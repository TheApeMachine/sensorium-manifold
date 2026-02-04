#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Manifold Physics Kernels
// =============================================================================
// Implements the core physics simulation for the thermo-manifold:
// - Particle-to-field scatter (gravity, heat)
// - Field-to-particle gather + integrated state update
// - Carrier-oscillator coupling
//
// Design principles:
// - Fused operations to minimize memory bandwidth
// - Hardware-accelerated trilinear interpolation via texture3d
// - All physics in one gather-update pass
// =============================================================================

// =============================================================================
// TEXTURE3D SAMPLER CONFIGURATION
// =============================================================================
// Metal's texture sampling hardware provides:
// - Free trilinear interpolation (in TMU, not ALU)
// - Automatic boundary clamping
// - Cache-optimized memory access patterns
//
// For field sampling, this gives ~2x speedup over manual interpolation.

// [CHOICE] texture sampler for periodic fields
// [FORMULA] address mode = repeat (torus / periodic domain)
// [REASON] particle dynamics use periodic boundaries; field sampling must match
// [NOTES] we keep a clamp sampler below for debugging/validation if needed.
constexpr sampler trilinear_sampler_periodic(
    coord::normalized,           // Use [0,1] normalized coordinates
    address::repeat,             // Periodic wrap at boundaries
    filter::linear,              // Trilinear interpolation
    mip_filter::none             // No mipmapping
);

// Clamp sampler (debug / exact edge behavior)
constexpr sampler trilinear_sampler_clamp(
    coord::normalized,
    address::clamp_to_edge,
    filter::linear,
    mip_filter::none
);

// Alternative sampler for nearest-neighbor (for debugging or exact cell access)
constexpr sampler nearest_sampler(
    coord::normalized,
    address::clamp_to_edge,
    filter::nearest
);

// -----------------------------------------------------------------------------
// Utility: Quiet NaN (fail-loudly sentinel)
// -----------------------------------------------------------------------------
// Metal does not guarantee `nanf()` is available; use a quiet-NaN bit pattern.
inline float qnan_f() {
    return as_type<float>(0x7FC00000u);
}

// -----------------------------------------------------------------------------
// Parameter Structs (must match ops.mm)
// -----------------------------------------------------------------------------

struct ManifoldFieldParams {
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    float grid_spacing;     // Physical size of each grid cell
    float inv_grid_spacing; // 1.0 / grid_spacing
};

struct ManifoldPhysicsParams {
    // Grid parameters
    uint32_t num_particles;
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    float grid_spacing;
    float inv_grid_spacing;
    float dt;                    // Time step
    
    // Fundamental physical constants (simulation units)
    float G;                     // Gravitational constant
    float k_B;                   // Boltzmann constant
    float sigma_SB;              // Stefan-Boltzmann constant
    float hbar;                  // Reduced Planck constant (ħ)
    
    // Material properties
    float particle_radius;       // Radius of particles
    float thermal_conductivity;  // k: heat transfer rate
    float specific_heat;         // c_v: heat capacity per unit mass  
    float dynamic_viscosity;     // η: resistance to flow
    float emissivity;            // ε: radiation efficiency (0-1)
    float young_modulus;         // E: collision stiffness
};

// =============================================================================
// Adaptive Thermodynamics: Fast reduction for global energy statistics
// =============================================================================
// We use a 2-pass reduction to compute:
//   mean_abs = mean(|x|), mean = mean(x), std = std(x)
// entirely on-GPU, so downstream kernels can do adaptive renormalization without
// CPU sync or "magic number" damping.
//
// Output format (single float4 in `out_stats`):
//   x: mean_abs
//   y: mean
//   z: std
//   w: count (as float)
//
// NOTE: Host must dispatch pass1 with exactly 256 threads/threadgroup and
//       num_threadgroups = ceil(N / 256).
// -----------------------------------------------------------------------------

// NOTE: Program-scope variables must reside in the constant address space in Metal.
constant uint kReduceThreads = 256;

kernel void reduce_float_stats_pass1(
    device const float* x           [[buffer(0)]],  // (N,)
    device float* group_stats       [[buffer(1)]],  // (num_groups * 4,) [sum_abs, sum, sum_sq, count]
    constant uint& N                [[buffer(2)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tg_id                      [[threadgroup_position_in_grid]]
) {
    uint idx = tg_id * kReduceThreads + tid;
    float v = (idx < N) ? x[idx] : 0.0f;
    float4 acc = float4(fabs(v), v, v * v, (idx < N) ? 1.0f : 0.0f);

    threadgroup float4 scratch[kReduceThreads];
    scratch[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction in shared memory.
    for (uint offset = kReduceThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            scratch[tid] += scratch[tid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        group_stats[tg_id * 4 + 0] = scratch[0].x;
        group_stats[tg_id * 4 + 1] = scratch[0].y;
        group_stats[tg_id * 4 + 2] = scratch[0].z;
        group_stats[tg_id * 4 + 3] = scratch[0].w;
    }
}

kernel void reduce_float_stats_finalize(
    device const float* group_stats [[buffer(0)]],  // (num_groups * 4,)
    device float* out_stats         [[buffer(1)]],  // (4,) [mean_abs, mean, std, count]
    constant uint& num_groups        [[buffer(2)]],
    uint tid                         [[thread_index_in_threadgroup]]
) {
    // One threadgroup (kReduceThreads) reduces all group_stats.
    float4 acc = float4(0.0f);
    for (uint i = tid; i < num_groups; i += kReduceThreads) {
        float sum_abs = group_stats[i * 4 + 0];
        float sum = group_stats[i * 4 + 1];
        float sum_sq = group_stats[i * 4 + 2];
        float count = group_stats[i * 4 + 3];
        acc += float4(sum_abs, sum, sum_sq, count);
    }

    threadgroup float4 scratch[kReduceThreads];
    scratch[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = kReduceThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            scratch[tid] += scratch[tid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float sum_abs = scratch[0].x;
        float sum = scratch[0].y;
        float sum_sq = scratch[0].z;
        float count = scratch[0].w;

        // [CHOICE] reduction empty-case semantics
        // [FORMULA] if count<=0: mean_abs=mean=std=0, count=0
        // [REASON] removes numerical clamp; makes empty reduction explicit
        if (!(count > 0.0f)) {
            out_stats[0] = 0.0f;
            out_stats[1] = 0.0f;
            out_stats[2] = 0.0f;
            out_stats[3] = 0.0f;
            return;
        }

        float mean_abs = sum_abs / count;
        float mean = sum / count;
        // [CHOICE] non-negative variance
        // [FORMULA] var = max(E[x^2] - E[x]^2, 0)
        // [REASON] rounding can produce tiny negative; project back to ℝ_{\ge 0}
        float var = max((sum_sq / count) - mean * mean, 0.0f);
        float std = sqrt(var);

        out_stats[0] = mean_abs;
        out_stats[1] = mean;
        out_stats[2] = std;
        out_stats[3] = count;
    }
}

// =============================================================================
// Stochastic helpers (hash + Box-Muller for N(0,1))
// =============================================================================
inline uint hash_u32(uint x) {
    // PCG-inspired mix (fast, decent avalanche)
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

inline float u01_from_u32(uint x) {
    // [CHOICE] uniform random in (0, 1] without eps-clamps
    // [FORMULA] u = (m + 1) / 2^24, where m ∈ [0, 2^24-1]
    // [REASON] avoids u=0 exactly (Box-Muller needs log(u))
    // [NOTES] allows u=1, which yields r=0 in Box-Muller (benign).
    uint m = (x & 0x00FFFFFFu);
    return (float)(m + 1u) * (1.0f / 16777216.0f);
}

inline float2 box_muller(float u1, float u2) {
    float r = sqrt(-2.0f * log(u1));
    float t = 2.0f * M_PI_F * u2;
    return float2(r * cos(t), r * sin(t));
}

inline float3 randn3(uint seed, uint idx) {
    // Deterministic per-(seed, idx) 3D standard normal.
    // Uses 6 uniforms derived from a hashed stream.
    uint s0 = hash_u32(seed ^ (idx * 0x9e3779b9u));
    uint s1 = hash_u32(s0 + 1u);
    uint s2 = hash_u32(s0 + 2u);
    uint s3 = hash_u32(s0 + 3u);
    float2 z0 = box_muller(u01_from_u32(s0), u01_from_u32(s1));
    float2 z1 = box_muller(u01_from_u32(s2), u01_from_u32(s3));
    // We only need 3 independent N(0,1) samples here.
    return float3(z0.x, z0.y, z1.x);
}

inline float2 randn2(uint seed, uint idx) {
    uint s0 = hash_u32(seed ^ (idx * 0x9e3779b9u));
    uint s1 = hash_u32(s0 + 1u);
    return box_muller(u01_from_u32(s0), u01_from_u32(s1));
}

inline float randn1(uint seed, uint idx) {
    return randn2(seed, idx).x;
}

// =============================================================================
// Spatial Hash Grid Structures (for O(N) collision detection)
// =============================================================================
// The spatial hash divides the simulation domain into cells. Each particle is
// assigned to a cell based on its position. Collision detection only checks
// particles in the same cell and 26 neighboring cells (3x3x3 neighborhood).
//
// Cell size should be >= 2 * particle_radius for correctness.
// For optimal performance, cell_size ≈ 2-4 * particle_radius.

struct SpatialHashParams {
    uint32_t num_particles;
    uint32_t grid_x;         // Number of cells in X
    uint32_t grid_y;         // Number of cells in Y
    uint32_t grid_z;         // Number of cells in Z
    float cell_size;         // Size of each cell
    float inv_cell_size;     // 1.0 / cell_size
    float domain_min_x;      // Domain minimum X
    float domain_min_y;      // Domain minimum Y
    float domain_min_z;      // Domain minimum Z
};

struct SpatialCollisionParams {
    uint32_t num_particles;
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    float cell_size;
    float inv_cell_size;
    float domain_min_x;
    float domain_min_y;
    float domain_min_z;
    float dt;
    float particle_radius;
    float young_modulus;
    float thermal_conductivity;
    float specific_heat;
    float restitution;
};

// Tiled scatter parameters for reduced atomic contention
struct TiledScatterParams {
    uint32_t num_particles;
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    float grid_spacing;
    float inv_grid_spacing;
    uint32_t tile_size;      // Particles per threadgroup tile
};

// -----------------------------------------------------------------------------
// Utility: Trilinear Interpolation
// -----------------------------------------------------------------------------

// Compute trilinear weights and grid indices for a position
inline void trilinear_coords(
    float3 pos,
    float inv_spacing,
    uint3 grid_dims,
    thread uint3& base_idx,
    thread float3& frac
) {
    // [CHOICE] periodic grid coordinate mapping
    // [FORMULA] g = (pos / Δx) mod grid_dims
    // [REASON] torus domain: positions and fields are periodic
    // [NOTES] This avoids non-physical boundary clamping artifacts.
    float3 g = pos * inv_spacing;
    float3 gd = float3(grid_dims);
    // Wrap into [0, grid_dims)
    g = g - gd * floor(g / gd);

    base_idx = uint3(floor(g));        // 0..dims-1
    frac = g - float3(base_idx);       // [0,1)
}

// Sample a 3D field with trilinear interpolation
inline float sample_field_trilinear(
    device const float* field,
    uint3 base_idx,
    float3 frac,
    uint3 grid_dims
) {
    // Compute strides
    uint stride_z = 1;
    uint stride_y = grid_dims.z;
    uint stride_x = grid_dims.y * grid_dims.z;

    // [CHOICE] periodic corner sampling
    // [FORMULA] (x1,y1,z1) = (x0+1,y0+1,z0+1) mod dims
    // [REASON] torus domain
    uint x0 = base_idx.x;
    uint y0 = base_idx.y;
    uint z0 = base_idx.z;
    uint x1 = (x0 + 1) % grid_dims.x;
    uint y1 = (y0 + 1) % grid_dims.y;
    uint z1 = (z0 + 1) % grid_dims.z;

    auto idx3 = [&](uint x, uint y, uint z) -> uint {
        return x * stride_x + y * stride_y + z * stride_z;
    };

    float c000 = field[idx3(x0, y0, z0)];
    float c001 = field[idx3(x0, y0, z1)];
    float c010 = field[idx3(x0, y1, z0)];
    float c011 = field[idx3(x0, y1, z1)];
    float c100 = field[idx3(x1, y0, z0)];
    float c101 = field[idx3(x1, y0, z1)];
    float c110 = field[idx3(x1, y1, z0)];
    float c111 = field[idx3(x1, y1, z1)];
    
    // Trilinear interpolation
    float fx = frac.x;
    float fy = frac.y;
    float fz = frac.z;
    
    float c00 = c000 * (1.0f - fz) + c001 * fz;
    float c01 = c010 * (1.0f - fz) + c011 * fz;
    float c10 = c100 * (1.0f - fz) + c101 * fz;
    float c11 = c110 * (1.0f - fz) + c111 * fz;
    
    float c0 = c00 * (1.0f - fy) + c01 * fy;
    float c1 = c10 * (1.0f - fy) + c11 * fy;
    
    return c0 * (1.0f - fx) + c1 * fx;
}

// Compute gradient of a 3D field at a position (central differences)
inline float3 sample_gradient_trilinear(
    device const float* field,
    uint3 base_idx,
    float3 frac,
    uint3 grid_dims,
    float inv_spacing
) {
    uint stride_z = 1;
    uint stride_y = grid_dims.z;
    uint stride_x = grid_dims.y * grid_dims.z;
    
    // Sample at offset positions for gradient
    // We approximate gradient using the interpolated values at slightly offset positions
    // For efficiency, we use the corner values to estimate gradient
    
    // Periodic corner sampling (same as sample_field_trilinear)
    uint x0 = base_idx.x;
    uint y0 = base_idx.y;
    uint z0 = base_idx.z;
    uint x1 = (x0 + 1) % grid_dims.x;
    uint y1 = (y0 + 1) % grid_dims.y;
    uint z1 = (z0 + 1) % grid_dims.z;

    auto idx3 = [&](uint x, uint y, uint z) -> uint {
        return x * stride_x + y * stride_y + z * stride_z;
    };

    float c000 = field[idx3(x0, y0, z0)];
    float c001 = field[idx3(x0, y0, z1)];
    float c010 = field[idx3(x0, y1, z0)];
    float c011 = field[idx3(x0, y1, z1)];
    float c100 = field[idx3(x1, y0, z0)];
    float c101 = field[idx3(x1, y0, z1)];
    float c110 = field[idx3(x1, y1, z0)];
    float c111 = field[idx3(x1, y1, z1)];
    
    // Gradient in each direction (using trilinear interpolation of face values)
    float fy = frac.y;
    float fz = frac.z;
    
    // dF/dx: difference between x=1 and x=0 faces
    float face_x0 = c000 * (1-fy) * (1-fz) + c010 * fy * (1-fz) + c001 * (1-fy) * fz + c011 * fy * fz;
    float face_x1 = c100 * (1-fy) * (1-fz) + c110 * fy * (1-fz) + c101 * (1-fy) * fz + c111 * fy * fz;
    float grad_x = (face_x1 - face_x0) * inv_spacing;
    
    float fx = frac.x;
    // dF/dy
    float face_y0 = c000 * (1-fx) * (1-fz) + c100 * fx * (1-fz) + c001 * (1-fx) * fz + c101 * fx * fz;
    float face_y1 = c010 * (1-fx) * (1-fz) + c110 * fx * (1-fz) + c011 * (1-fx) * fz + c111 * fx * fz;
    float grad_y = (face_y1 - face_y0) * inv_spacing;
    
    // dF/dz
    float face_z0 = c000 * (1-fx) * (1-fy) + c100 * fx * (1-fy) + c010 * (1-fx) * fy + c110 * fx * fy;
    float face_z1 = c001 * (1-fx) * (1-fy) + c101 * fx * (1-fy) + c011 * (1-fx) * fy + c111 * fx * fy;
    float grad_z = (face_z1 - face_z0) * inv_spacing;
    
    return float3(grad_x, grad_y, grad_z);
}

// -----------------------------------------------------------------------------
// Kernel: Scatter particles to fields (gravity + heat)
// -----------------------------------------------------------------------------
// Each particle contributes its mass to the gravity field and its heat to the
// temperature field using trilinear interpolation weights.

// Per-particle scatter (one thread per particle)
kernel void scatter_particle(
    device const float* particle_pos      [[buffer(0)]],  // N * 3
    device const float* particle_mass     [[buffer(1)]],  // N
    device const float* particle_heat     [[buffer(2)]],  // N
    device const float* particle_energy   [[buffer(3)]],  // N (oscillator/internal mode energy)
    device atomic_float* gravity_field    [[buffer(4)]],  // X * Y * Z
    device atomic_float* heat_field       [[buffer(5)]],  // X * Y * Z (total internal energy per cell)
    constant ManifoldPhysicsParams& p     [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.num_particles) return;
    
    // Read particle position
    float3 pos = float3(
        particle_pos[gid * 3 + 0],
        particle_pos[gid * 3 + 1],
        particle_pos[gid * 3 + 2]
    );
    float mass = particle_mass[gid];
    float heat = particle_heat[gid];
    float e_osc = particle_energy[gid];
    float e_total = heat + e_osc;
    
    // Get trilinear coordinates
    uint3 base_idx;
    float3 frac;
    uint3 grid_dims = uint3(p.grid_x, p.grid_y, p.grid_z);
    trilinear_coords(pos, p.inv_grid_spacing, grid_dims, base_idx, frac);
    
    // Compute 8 trilinear weights
    float wx0 = 1.0f - frac.x, wx1 = frac.x;
    float wy0 = 1.0f - frac.y, wy1 = frac.y;
    float wz0 = 1.0f - frac.z, wz1 = frac.z;
    
    float weights[8] = {
        wx0 * wy0 * wz0,  // 000
        wx0 * wy0 * wz1,  // 001
        wx0 * wy1 * wz0,  // 010
        wx0 * wy1 * wz1,  // 011
        wx1 * wy0 * wz0,  // 100
        wx1 * wy0 * wz1,  // 101
        wx1 * wy1 * wz0,  // 110
        wx1 * wy1 * wz1   // 111
    };
    
    // [CHOICE] periodic scatter to grid corners
    // [FORMULA] deposit into 8 corners with (x1,y1,z1) wrapped mod dims
    // [REASON] torus domain must conserve mass/heat across boundaries
    // [NOTES] avoids “edge sinks” created by clamping.
    uint gx = p.grid_x, gy = p.grid_y, gz = p.grid_z;
    uint x0 = base_idx.x;
    uint y0 = base_idx.y;
    uint z0 = base_idx.z;
    uint x1 = (x0 + 1) % gx;
    uint y1 = (y0 + 1) % gy;
    uint z1 = (z0 + 1) % gz;

    uint stride_z = 1;
    uint stride_y = gz;
    uint stride_x = gy * gz;
    auto idx3 = [&](uint x, uint y, uint z) -> uint {
        return x * stride_x + y * stride_y + z * stride_z;
    };

    uint idxs[8] = {
        idx3(x0, y0, z0),
        idx3(x0, y0, z1),
        idx3(x0, y1, z0),
        idx3(x0, y1, z1),
        idx3(x1, y0, z0),
        idx3(x1, y0, z1),
        idx3(x1, y1, z0),
        idx3(x1, y1, z1),
    };

    for (int i = 0; i < 8; i++) {
        uint idx = idxs[i];
        atomic_fetch_add_explicit(&gravity_field[idx], mass * weights[i], memory_order_relaxed);
        // [CHOICE] total internal energy deposition
        // [FORMULA] Q_cell := Σ_i w_i (Q_i + E_osc,i)
        // [REASON] temperature is defined from total internal energy, not thermal Q alone
        atomic_fetch_add_explicit(&heat_field[idx], e_total * weights[i], memory_order_relaxed);
    }
}

// -----------------------------------------------------------------------------
// Kernel: Gather from fields + update particle state (FUSED)
// -----------------------------------------------------------------------------
// This is the main physics kernel. For each particle:
// 1. Sample gravity potential → compute force
// 2. Sample temperature → compute heat exchange
// 3. Update velocity from force (with viscosity)
// 4. Update position from velocity
// 5. Update energy, heat, excitation from thermodynamic rules
// All in one kernel, one read per field, minimal memory traffic.

kernel void gather_update_particles(
    // Fields (read-only)
    device const float* gravity_potential [[buffer(0)]],  // X * Y * Z
    device const float* temperature_field [[buffer(1)]],  // X * Y * Z
    device const float* mass_field        [[buffer(2)]],  // X * Y * Z (scattered mass-per-cell)
    // Particle state (read-write)
    device float* particle_pos            [[buffer(3)]],  // N * 3
    device float* particle_vel            [[buffer(4)]],  // N * 3
    device float* particle_energy         [[buffer(5)]],  // N
    device float* particle_heat           [[buffer(6)]],  // N
    device float* particle_excitation     [[buffer(7)]],  // N
    device const float* particle_mass     [[buffer(8)]],  // N (read-only, doesn't change)
    // Parameters
    constant ManifoldPhysicsParams& p     [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.num_particles) return;
    
    // -------------------------------------------------------------------------
    // 1. Read current particle state
    // -------------------------------------------------------------------------
    float3 pos = float3(
        particle_pos[gid * 3 + 0],
        particle_pos[gid * 3 + 1],
        particle_pos[gid * 3 + 2]
    );
    float3 vel = float3(
        particle_vel[gid * 3 + 0],
        particle_vel[gid * 3 + 1],
        particle_vel[gid * 3 + 2]
    );
    float energy = particle_energy[gid];
    float heat = particle_heat[gid];
    float excitation = particle_excitation[gid];
    float mass = particle_mass[gid];

    // [CHOICE] fundamental invariants (fail loudly)
    // [FORMULA] require: m>0, Δx>0, r>0, c_v>0
    // [REASON] these are physical preconditions; silent clamps hide invalid states
    // [NOTES] on violation we write NaNs to state to surface errors immediately.
    if (!(mass > 0.0f) || !(p.grid_spacing > 0.0f) || !(p.particle_radius > 0.0f) || !(p.specific_heat > 0.0f)) {
        float qn = qnan_f();
        particle_pos[gid * 3 + 0] = qn;
        particle_pos[gid * 3 + 1] = qn;
        particle_pos[gid * 3 + 2] = qn;
        particle_vel[gid * 3 + 0] = qn;
        particle_vel[gid * 3 + 1] = qn;
        particle_vel[gid * 3 + 2] = qn;
        particle_energy[gid] = qn;
        particle_heat[gid] = qn;
        particle_excitation[gid] = qn;
        return;
    }
    
    // -------------------------------------------------------------------------
    // 2. Gather from fields (trilinear interpolation)
    // -------------------------------------------------------------------------
    uint3 base_idx;
    float3 frac;
    uint3 grid_dims = uint3(p.grid_x, p.grid_y, p.grid_z);
    trilinear_coords(pos, p.inv_grid_spacing, grid_dims, base_idx, frac);
    
    // [CHOICE] gravity coupling (particle ← potential field)
    // [FORMULA] Poisson: ∇²φ = 4πGρ  =>  acceleration a = -∇φ
    //           Force:   F = m a = -m ∇φ
    // [REASON] standard Newtonian gravity in potential form
    // [NOTES] We choose to include G in the Poisson solve (φ already includes G).
    //         Therefore we MUST NOT multiply by G again here.
    float3 gravity_grad = sample_gradient_trilinear(
        gravity_potential, base_idx, frac, grid_dims, p.inv_grid_spacing
    );
    float3 gravity_force = -gravity_grad * mass;
    
    // =========================================================================
    // TEMPERATURE AND PRESSURE
    // =========================================================================
    // Sample local temperature field (this is T from heat diffusion on grid)
    float local_temp = sample_field_trilinear(
        temperature_field, base_idx, frac, grid_dims
    );
    
    // [CHOICE] particle internal temperature proxy
    // [FORMULA] T_i = Q_i / (m_i c_v)
    // [REASON] internal-energy-to-temperature mapping for a lumped particle
    // [NOTES] invariant m>0 and c_v>0 is enforced above (no silent clamp).
    // [CHOICE] particle temperature from total internal energy
    // [FORMULA] T_i = (Q_i + E_osc,i) / (m_i c_v)
    // [REASON] closes bookkeeping between thermal + oscillator energy stores
    float particle_temp = (heat + energy) / (mass * p.specific_heat);
    
    // [CHOICE] pressure force (continuum, ideal-gas EOS)
    // [FORMULA] EOS:     P = ρ k_B T
    //          Force:   a = -(1/ρ) ∇P
    //                  ∇P = k_B (T ∇ρ + ρ ∇T)
    // [REASON] replaces the prior heuristic with a dimensionally consistent continuum form
    // [NOTES] Vacuum semantics: if ρ<=0, pressure contribution is defined as 0 (no clamp-as-physics).
    float3 temp_grad = sample_gradient_trilinear(
        temperature_field, base_idx, frac, grid_dims, p.inv_grid_spacing
    );

    // Local density from scattered mass field: ρ = m_cell / h^3
    float m_cell = sample_field_trilinear(mass_field, base_idx, frac, grid_dims);
    float3 grad_m = sample_gradient_trilinear(mass_field, base_idx, frac, grid_dims, p.inv_grid_spacing);
    float h = p.grid_spacing;
    // invariant Δx>0 enforced above.
    float inv_h3 = 1.0f / (h * h * h);
    float rho = m_cell * inv_h3;
    float3 grad_rho = grad_m * inv_h3;

    float3 pressure_force = float3(0.0f);
    if (rho > 0.0f) {
        float3 gradP = p.k_B * (local_temp * grad_rho + rho * temp_grad);
        float3 aP = -gradP / rho;
        pressure_force = aP * mass;
    }
    
    // =========================================================================
    // HEAT TRANSFER: Newton's law of cooling + Stefan-Boltzmann radiation
    // =========================================================================
    // [CHOICE] conduction to ambient medium (sphere in infinite medium)
    // [FORMULA] Q̇ = 4π κ r (T_env - T_particle)
    // [REASON] steady-state conduction solution of Laplace equation around a sphere
    // [NOTES] κ here is the medium thermal conductivity (not diffusivity).
    float r = p.particle_radius;
    float dQ_conduction = (4.0f * M_PI_F) * p.thermal_conductivity * r * (local_temp - particle_temp) * p.dt;
    heat += dQ_conduction;
    
    // [CHOICE] radiative cooling
    // [FORMULA] P = ε σ A T^4, with A = 4π r²
    // [REASON] blackbody radiation loss from particle surface
    // [NOTES] No absorbed constants: σ is the (unit-system converted) universal constant.
    float surface_area = 4.0f * M_PI_F * r * r;
    float T4 = particle_temp * particle_temp * particle_temp * particle_temp;
    float dQ_radiation = p.emissivity * p.sigma_SB * surface_area * T4 * p.dt;
    // [CHOICE] non-negative thermal energy (0 K baseline)
    // [FORMULA] Q >= 0
    // [REASON] internal thermal energy relative to absolute zero cannot be negative
    // [NOTES] this is a physical boundary condition, not a tuneable clamp.
    heat = max(heat - dQ_radiation, 0.0f);
    
    // Update temperature after heat exchange
    particle_temp = (heat + energy) / (mass * p.specific_heat);

    // =========================================================================
    // THERMAL ↔ OSCILLATOR ENERGY EXCHANGE (local, conservative)
    // =========================================================================
    // [CHOICE] equilibrium oscillator energy (quantum harmonic oscillator)
    // [FORMULA] E_eq(ω,T) = ħω / (exp(ħω/(k_B T)) - 1)
    // [REASON] removes ad-hoc cutoffs; recovers classical limit at high T / low ω
    // [NOTES] - Uses ω = |excitation| as an intrinsic mode frequency.
    //         - Exchange conserves (Q + E_osc) locally: heat -= ΔE, energy += ΔE.
    //         - Timescale derived from existing conduction coefficient (no new knobs).
    {
        float kappa = p.thermal_conductivity;
        float cv = p.specific_heat;
        float omega = fabs(excitation);

        // Natural thermalization timescale implied by conduction to the medium:
        // τ = (m c_v) / (4π κ r)
        float denom_tau = (4.0f * M_PI_F) * kappa * r;
        float tau = (denom_tau > 0.0f) ? (mass * cv) / denom_tau : INFINITY;

        // α = 1 - exp(-dt/τ)
        float alpha = (isfinite(tau) && tau > 0.0f) ? (1.0f - exp(-p.dt / tau)) : 0.0f;

        float T = max(particle_temp, 0.0f);
        float E_eq = 0.0f;
        float kBT = p.k_B * T;
        if (kBT > 0.0f && omega > 0.0f && p.hbar > 0.0f) {
            float x = (p.hbar * omega) / kBT;
            // Numerical safety: for large x, exp(x) overflows and E_eq -> 0.
            if (x > 80.0f) {
                E_eq = 0.0f;
            } else if (x < 1.0e-4f) {
                // Classical limit: ħω/(exp(x)-1) ≈ ħω/x = k_B T.
                E_eq = kBT;
            } else {
                float denom = exp(x) - 1.0f;
                E_eq = (denom > 0.0f) ? (p.hbar * omega) / denom : 0.0f;
            }
        } else if (kBT > 0.0f && omega <= 0.0f) {
            // ω→0 limit: E_eq -> k_B T.
            E_eq = kBT;
        }

        float dE = alpha * (E_eq - energy);
        // Prevent drawing more thermal energy than available (Q >= 0 boundary).
        if (dE > 0.0f) {
            dE = min(dE, heat);
        }
        energy += dE;
        heat -= dE;
    }
    
    // =========================================================================
    // EXCITATION DYNAMICS (oscillator frequency)
    // =========================================================================
    // Excitation represents the oscillator's INTRINSIC natural frequency.
    // This is a conserved property that does NOT change during simulation.
    // 
    // The frequency is set at particle creation and remains fixed.
    // This ensures spectral diversity is preserved throughout the simulation.
    //
    // Note: We do NOT modify excitation here - it's an intrinsic property.
    // The spectral carrier layer reads excitation as oscillator frequency (ω).
    //
    // NOTE: We intentionally do NOT apply any additional "energy -> heat"
    // transfer here beyond physically modeled mechanisms (drag dissipation,
    // collisions, conduction/radiation). This avoids ad-hoc damping terms.
    energy = max(energy, 0.0f);
    heat = max(heat, 0.0f);
    
    // =========================================================================
    // SYMPLECTIC INTEGRATION (Velocity Verlet on conservative forces)
    // =========================================================================
    // We split dynamics into:
    // - Conservative forces (gravity + pressure): integrate with Velocity Verlet
    // - Dissipative drag: applied as an exact exponential half-step (Strang split)
    //
    // Stokes drag force: F_drag = -gamma v, gamma = 6π μ(T) r
    // Acceleration contribution: a_drag = -(gamma/m) v
    //
    // Ideal-gas-inspired dynamic viscosity scaling (kinetic theory):
    //   μ(T) ∝ sqrt(T)
    // We treat `p.dynamic_viscosity` as μ_ref at T_ref = 1 (dimensionless).
    // [CHOICE] viscosity temperature dependence (physical domain constraint)
    // [FORMULA] μ(T) ∝ sqrt(T), defined for T >= 0
    // [REASON] kinetic-theory-inspired scaling used by the drag model
    // [NOTES] - Physical temperature cannot be negative; we enforce T>=0 as a boundary.
    //         - Spectral diffusion on fp32 can introduce tiny negative values from rounding
    //           even when the true solution is nonnegative. Projecting to T>=0 avoids
    //           cascading NaNs without introducing a new tunable threshold.
    //         - Non-finite temperature (NaN/inf) remains a hard error (fail loudly).
    if (!isfinite(local_temp)) {
        float qn = qnan_f();
        particle_vel[gid * 3 + 0] = qn;
        particle_vel[gid * 3 + 1] = qn;
        particle_vel[gid * 3 + 2] = qn;
        particle_heat[gid] = qn;
        return;
    }
    local_temp = max(local_temp, 0.0f);
    float mu = p.dynamic_viscosity * sqrt(local_temp);
    float gamma = 6.0f * 3.14159f * mu * r;
    float inv_m = 1.0f / mass;
    float dt = p.dt;
    float half_dt = 0.5f * dt;
    float drag_half = exp(-(gamma * inv_m) * half_dt);

    // Conservative acceleration at x(t)
    float3 a0 = (gravity_force + pressure_force) * inv_m;

    // ---- Drag half-step (exact), convert KE loss -> heat
    float ke0 = 0.5f * mass * dot(vel, vel);
    vel *= drag_half;
    float ke1 = 0.5f * mass * dot(vel, vel);
    heat += max(ke0 - ke1, 0.0f);

    // ---- Kick (half): v(t+dt/2) = v + (dt/2) a(x_t)
    vel += a0 * half_dt;

    // ---- Drift: x(t+dt) = x + dt * v(t+dt/2)
    pos += vel * dt;
    
    // Periodic boundary conditions (torus domain):
    // This removes wall reflections/fudges and is a standard physically
    // interpretable choice for finite domains approximating "unbounded" space.
    float3 domain = float3(p.grid_x, p.grid_y, p.grid_z) * p.grid_spacing;
    // Wrap into [0, domain)
    pos = pos - domain * floor(pos / domain);

    // Conservative acceleration at x(t+dt) (same fields; second force evaluation)
    {
        uint3 base2;
        float3 frac2;
        trilinear_coords(pos, p.inv_grid_spacing, grid_dims, base2, frac2);
        float3 ggrad2 = sample_gradient_trilinear(gravity_potential, base2, frac2, grid_dims, p.inv_grid_spacing);
        float3 tgrad2 = sample_gradient_trilinear(temperature_field, base2, frac2, grid_dims, p.inv_grid_spacing);
        float3 Fg2 = -ggrad2 * mass * p.G;
        float3 Fp2 = -tgrad2 * p.k_B * mass;
        float3 a1 = (Fg2 + Fp2) * inv_m;

        // ---- Kick (half): v(t+dt) = v(t+dt/2) + (dt/2) a(x_{t+dt})
        vel += a1 * half_dt;
    }

    // ---- Drag half-step (exact), convert KE loss -> heat
    float ke2 = 0.5f * mass * dot(vel, vel);
    vel *= drag_half;
    float ke3 = 0.5f * mass * dot(vel, vel);
    heat += max(ke2 - ke3, 0.0f);
    
    // -------------------------------------------------------------------------
    // 6. Write back updated state
    // -------------------------------------------------------------------------
    particle_pos[gid * 3 + 0] = pos.x;
    particle_pos[gid * 3 + 1] = pos.y;
    particle_pos[gid * 3 + 2] = pos.z;
    
    particle_vel[gid * 3 + 0] = vel.x;
    particle_vel[gid * 3 + 1] = vel.y;
    particle_vel[gid * 3 + 2] = vel.z;
    
    particle_energy[gid] = energy;
    particle_heat[gid] = heat;
    particle_excitation[gid] = excitation;
}

// -----------------------------------------------------------------------------
// Kernel: Gather + BAOAB Langevin update + Kuramoto mean-field "entanglement"
// -----------------------------------------------------------------------------
// Goal:
// - Replace clamp-heavy semi-Euler update with a stable Langevin integrator.
// - Add a non-local coupling signal via carrier mean-field alignment.
//
// Notes:
// - This does NOT require backprop. The "loss" is the effective energy landscape
//   implied by forces + thermostat.
// - We keep the legacy kernel intact; this is opt-in from the host.
//
// BAOAB splitting (one step):
//   v <- v + (dt/2m) F(x)                          (B)
//   v <- a v + b N(0,1)  with a=exp(-γdt), b=...   (A/O)
//   x <- x + dt v                                  (O)
//   v <- v + (dt/2m) F(x_new)                      (B)
//
// Kuramoto mean-field:
// - g_i = Σ_k T(ω_i, ω_k, σ_k) * c_k
// - alignment = Re(e^{-iφ_i} g_i) / (|g_i|+eps) ∈ [-1,1]
// - entangle_force ∥ ∇φ_gravity, scaled by alignment (global, non-local signal)
//
// -----------------------------------------------------------------------------
// Kernel: Gather from fields using TEXTURE3D (hardware-accelerated)
// -----------------------------------------------------------------------------
// This version uses Metal's texture sampling hardware for:
// - Free trilinear interpolation in texture unit (TMU)
// - Cache-optimized memory access patterns
// - ~2x faster than manual buffer interpolation
//
// To use this kernel, the host must:
// 1. Create MTLTexture objects with MTLPixelFormatR32Float
// 2. Copy field data into textures before each frame
// 3. Use MTLSamplerDescriptor with linear filtering

struct ManifoldPhysicsTextureParams {
    uint32_t num_particles;
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    float grid_spacing;
    float inv_grid_spacing;
    float dt;
    float G;
    float k_B;
    float sigma_SB;
    float particle_radius;
    float thermal_conductivity;
    float specific_heat;
    float dynamic_viscosity;
    float emissivity;
    float young_modulus;
};

kernel void gather_update_particles_textured(
    // Fields as 3D textures (read-only, hardware trilinear)
    texture3d<float, access::sample> gravity_potential [[texture(0)]],
    texture3d<float, access::sample> temperature_field [[texture(1)]],
    // Particle state (read-write)
    device float* particle_pos            [[buffer(0)]],  // N * 3
    device float* particle_vel            [[buffer(1)]],  // N * 3
    device float* particle_energy         [[buffer(2)]],  // N
    device float* particle_heat           [[buffer(3)]],  // N
    device float* particle_excitation     [[buffer(4)]],  // N
    device const float* particle_mass     [[buffer(5)]],  // N
    // Parameters
    constant ManifoldPhysicsTextureParams& p [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.num_particles) return;
    
    // Read current particle state
    float3 pos = float3(
        particle_pos[gid * 3 + 0],
        particle_pos[gid * 3 + 1],
        particle_pos[gid * 3 + 2]
    );
    float3 vel = float3(
        particle_vel[gid * 3 + 0],
        particle_vel[gid * 3 + 1],
        particle_vel[gid * 3 + 2]
    );
    float energy = particle_energy[gid];
    float heat = particle_heat[gid];
    float excitation = particle_excitation[gid];
    float mass = particle_mass[gid];

    // Fundamental invariants (fail loudly). See buffer kernel for rationale.
    if (!(mass > 0.0f) || !(p.grid_spacing > 0.0f) || !(p.particle_radius > 0.0f) || !(p.specific_heat > 0.0f)) {
        float qn = qnan_f();
        particle_pos[gid * 3 + 0] = qn;
        particle_pos[gid * 3 + 1] = qn;
        particle_pos[gid * 3 + 2] = qn;
        particle_vel[gid * 3 + 0] = qn;
        particle_vel[gid * 3 + 1] = qn;
        particle_vel[gid * 3 + 2] = qn;
        particle_energy[gid] = qn;
        particle_heat[gid] = qn;
        particle_excitation[gid] = qn;
        return;
    }
    
    // Convert position to normalized texture coordinates [0, 1]
    float3 grid_dims = float3(p.grid_x, p.grid_y, p.grid_z);
    float3 tex_coord = pos * p.inv_grid_spacing / grid_dims;
    
    // Sample temperature field using hardware trilinear interpolation
    // Note: texture.sample() returns float4, we use .x for single-channel
    // (gravity potential is only used for gradient, not the value itself)
    float local_temp = temperature_field.sample(trilinear_sampler_periodic, tex_coord).x;
    
    // Compute gradients using finite differences on texture samples
    float3 texel_size = 1.0f / grid_dims;
    float3 gravity_grad = float3(
        gravity_potential.sample(trilinear_sampler_periodic, tex_coord + float3(texel_size.x, 0, 0)).x -
        gravity_potential.sample(trilinear_sampler_periodic, tex_coord - float3(texel_size.x, 0, 0)).x,
        gravity_potential.sample(trilinear_sampler_periodic, tex_coord + float3(0, texel_size.y, 0)).x -
        gravity_potential.sample(trilinear_sampler_periodic, tex_coord - float3(0, texel_size.y, 0)).x,
        gravity_potential.sample(trilinear_sampler_periodic, tex_coord + float3(0, 0, texel_size.z)).x -
        gravity_potential.sample(trilinear_sampler_periodic, tex_coord - float3(0, 0, texel_size.z)).x
    ) * (0.5f * p.inv_grid_spacing);
    
    float3 temp_grad = float3(
        temperature_field.sample(trilinear_sampler_periodic, tex_coord + float3(texel_size.x, 0, 0)).x -
        temperature_field.sample(trilinear_sampler_periodic, tex_coord - float3(texel_size.x, 0, 0)).x,
        temperature_field.sample(trilinear_sampler_periodic, tex_coord + float3(0, texel_size.y, 0)).x -
        temperature_field.sample(trilinear_sampler_periodic, tex_coord - float3(0, texel_size.y, 0)).x,
        temperature_field.sample(trilinear_sampler_periodic, tex_coord + float3(0, 0, texel_size.z)).x -
        temperature_field.sample(trilinear_sampler_periodic, tex_coord - float3(0, 0, texel_size.z)).x
    ) * (0.5f * p.inv_grid_spacing);
    
    // [CHOICE] gravity coupling (textured gather)
    // [FORMULA] Poisson: ∇²φ = 4πGρ  =>  a = -∇φ ; F = -m ∇φ
    // [REASON] match buffer version + avoid G double-counting
    // [NOTES] φ already includes G via Poisson solve.
    float3 gravity_force = -gravity_grad * mass;
    float particle_temp = heat / (mass * p.specific_heat);
    float3 pressure_force = -temp_grad * p.k_B * mass;
    
    // Heat transfer
    float r = p.particle_radius;
    // Conduction to ambient (see buffer kernel for choice block).
    float dQ_conduction = (4.0f * M_PI_F) * p.thermal_conductivity * r * (local_temp - particle_temp) * p.dt;
    heat += dQ_conduction;
    
    // Radiative cooling (same as buffer version; see choice block there).
    float surface_area = 4.0f * M_PI_F * r * r;
    float T4 = particle_temp * particle_temp * particle_temp * particle_temp;
    float dQ_radiation = p.emissivity * p.sigma_SB * surface_area * T4 * p.dt;
    heat = max(heat - dQ_radiation, 0.0f);
    
    particle_temp = heat / (mass * p.specific_heat);
    
    // Excitation is an intrinsic property - do NOT modify it.
    // See gather_update_particles for documentation.
    
    // NOTE: No ad-hoc "energy -> heat" thermalization term here. See
    // gather_update_particles for rationale.
    energy = max(energy, 0.0f);
    heat = max(heat, 0.0f);
    
    // =========================================================================
    // SYMPLECTIC INTEGRATION (Velocity Verlet on conservative forces)
    // =========================================================================
    // Temperature-dependent viscosity (ideal-gas-inspired): μ(T) ∝ sqrt(T).
    // See buffer kernel for domain/boundary rationale.
    if (!isfinite(local_temp)) {
        float qn = qnan_f();
        particle_vel[gid * 3 + 0] = qn;
        particle_vel[gid * 3 + 1] = qn;
        particle_vel[gid * 3 + 2] = qn;
        particle_heat[gid] = qn;
        return;
    }
    local_temp = max(local_temp, 0.0f);
    float mu = p.dynamic_viscosity * sqrt(local_temp);
    float gamma = 6.0f * 3.14159f * mu * r;
    float inv_m = 1.0f / mass;
    float dt = p.dt;
    float half_dt = 0.5f * dt;
    float drag_half = exp(-(gamma * inv_m) * half_dt);

    float3 a0 = (gravity_force + pressure_force) * inv_m;

    // Drag half-step (exact), KE loss -> heat
    float ke0 = 0.5f * mass * dot(vel, vel);
    vel *= drag_half;
    float ke1 = 0.5f * mass * dot(vel, vel);
    heat += max(ke0 - ke1, 0.0f);

    // Kick (half)
    vel += a0 * half_dt;

    // Drift
    pos += vel * dt;

    // Periodic boundary conditions (torus domain). See buffer version.
    float3 domain = float3(p.grid_x, p.grid_y, p.grid_z) * p.grid_spacing;
    pos = pos - domain * floor(pos / domain);

    // Recompute gradients at new position using texture samples
    float3 grid_dims2 = float3(p.grid_x, p.grid_y, p.grid_z);
    float3 tex_coord2 = pos * p.inv_grid_spacing / grid_dims2;
    float3 texel_size2 = 1.0f / grid_dims2;

    float3 gravity_grad2 = float3(
        gravity_potential.sample(trilinear_sampler_periodic, tex_coord2 + float3(texel_size2.x, 0, 0)).x -
        gravity_potential.sample(trilinear_sampler_periodic, tex_coord2 - float3(texel_size2.x, 0, 0)).x,
        gravity_potential.sample(trilinear_sampler_periodic, tex_coord2 + float3(0, texel_size2.y, 0)).x -
        gravity_potential.sample(trilinear_sampler_periodic, tex_coord2 - float3(0, texel_size2.y, 0)).x,
        gravity_potential.sample(trilinear_sampler_periodic, tex_coord2 + float3(0, 0, texel_size2.z)).x -
        gravity_potential.sample(trilinear_sampler_periodic, tex_coord2 - float3(0, 0, texel_size2.z)).x
    ) * (0.5f * p.inv_grid_spacing);

    float3 temp_grad2 = float3(
        temperature_field.sample(trilinear_sampler_periodic, tex_coord2 + float3(texel_size2.x, 0, 0)).x -
        temperature_field.sample(trilinear_sampler_periodic, tex_coord2 - float3(texel_size2.x, 0, 0)).x,
        temperature_field.sample(trilinear_sampler_periodic, tex_coord2 + float3(0, texel_size2.y, 0)).x -
        temperature_field.sample(trilinear_sampler_periodic, tex_coord2 - float3(0, texel_size2.y, 0)).x,
        temperature_field.sample(trilinear_sampler_periodic, tex_coord2 + float3(0, 0, texel_size2.z)).x -
        temperature_field.sample(trilinear_sampler_periodic, tex_coord2 - float3(0, 0, texel_size2.z)).x
    ) * (0.5f * p.inv_grid_spacing);

    float3 Fg2 = -gravity_grad2 * mass * p.G;
    float3 Fp2 = -temp_grad2 * p.k_B * mass;
    float3 a1 = (Fg2 + Fp2) * inv_m;

    // Kick (half)
    vel += a1 * half_dt;

    // Drag half-step (exact), KE loss -> heat
    float ke2 = 0.5f * mass * dot(vel, vel);
    vel *= drag_half;
    float ke3 = 0.5f * mass * dot(vel, vel);
    heat += max(ke2 - ke3, 0.0f);
    
    // Write back
    particle_pos[gid * 3 + 0] = pos.x;
    particle_pos[gid * 3 + 1] = pos.y;
    particle_pos[gid * 3 + 2] = pos.z;
    
    particle_vel[gid * 3 + 0] = vel.x;
    particle_vel[gid * 3 + 1] = vel.y;
    particle_vel[gid * 3 + 2] = vel.z;
    
    particle_energy[gid] = energy;
    particle_heat[gid] = heat;
    particle_excitation[gid] = excitation;
}

// -----------------------------------------------------------------------------
// Kernel: Heat diffusion on the field (Laplacian stencil)
// -----------------------------------------------------------------------------
// Evolves the temperature field via diffusion: dT/dt = α ∇²T
// Uses periodic boundary conditions (torus) to match particle dynamics.

kernel void diffuse_heat_field(
    device const float* temp_in           [[buffer(0)]],  // X * Y * Z
    device float* temp_out                [[buffer(1)]],  // X * Y * Z
    constant ManifoldFieldParams& p       [[buffer(2)]],
    constant float& diffusion_coef        [[buffer(3)]],
    constant float& dt                    [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= p.grid_x || gid.y >= p.grid_y || gid.z >= p.grid_z) return;
    
    uint stride_z = 1;
    uint stride_y = p.grid_z;
    uint stride_x = p.grid_y * p.grid_z;
    uint idx = gid.x * stride_x + gid.y * stride_y + gid.z * stride_z;
    
    float center = temp_in[idx];
    
    // 6-point Laplacian stencil with PERIODIC boundary handling
    uint x_prev = (gid.x == 0) ? p.grid_x - 1 : gid.x - 1;
    uint x_next = (gid.x == p.grid_x - 1) ? 0 : gid.x + 1;
    uint y_prev = (gid.y == 0) ? p.grid_y - 1 : gid.y - 1;
    uint y_next = (gid.y == p.grid_y - 1) ? 0 : gid.y + 1;
    uint z_prev = (gid.z == 0) ? p.grid_z - 1 : gid.z - 1;
    uint z_next = (gid.z == p.grid_z - 1) ? 0 : gid.z + 1;

    float xm = temp_in[x_prev * stride_x + gid.y * stride_y + gid.z * stride_z];
    float xp = temp_in[x_next * stride_x + gid.y * stride_y + gid.z * stride_z];
    float ym = temp_in[gid.x * stride_x + y_prev * stride_y + gid.z * stride_z];
    float yp = temp_in[gid.x * stride_x + y_next * stride_y + gid.z * stride_z];
    float zm = temp_in[gid.x * stride_x + gid.y * stride_y + z_prev * stride_z];
    float zp = temp_in[gid.x * stride_x + gid.y * stride_y + z_next * stride_z];
    
    float laplacian = (xm + xp + ym + yp + zm + zp - 6.0f * center) 
                      * (p.inv_grid_spacing * p.inv_grid_spacing);
    
    temp_out[idx] = center + diffusion_coef * laplacian * dt;
}

// -----------------------------------------------------------------------------
// Kernel: Solve Poisson equation for gravity (Jacobi iteration step)
// -----------------------------------------------------------------------------
// ∇²φ = 4πG ρ  →  One Jacobi iteration step
// Uses periodic boundary conditions (torus) to match particle dynamics.

kernel void poisson_jacobi_step(
    device const float* phi_in            [[buffer(0)]],  // X * Y * Z (potential)
    device const float* rho               [[buffer(1)]],  // X * Y * Z (density/mass)
    device float* phi_out                 [[buffer(2)]],  // X * Y * Z
    constant ManifoldFieldParams& p       [[buffer(3)]],
    constant float& gravity_4pi           [[buffer(4)]],  // 4 * pi * G
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= p.grid_x || gid.y >= p.grid_y || gid.z >= p.grid_z) return;
    
    uint stride_z = 1;
    uint stride_y = p.grid_z;
    uint stride_x = p.grid_y * p.grid_z;
    uint idx = gid.x * stride_x + gid.y * stride_y + gid.z * stride_z;
    
    // Periodic boundary handling
    uint x_prev = (gid.x == 0) ? p.grid_x - 1 : gid.x - 1;
    uint x_next = (gid.x == p.grid_x - 1) ? 0 : gid.x + 1;
    uint y_prev = (gid.y == 0) ? p.grid_y - 1 : gid.y - 1;
    uint y_next = (gid.y == p.grid_y - 1) ? 0 : gid.y + 1;
    uint z_prev = (gid.z == 0) ? p.grid_z - 1 : gid.z - 1;
    uint z_next = (gid.z == p.grid_z - 1) ? 0 : gid.z + 1;

    float xm = phi_in[x_prev * stride_x + gid.y * stride_y + gid.z * stride_z];
    float xp = phi_in[x_next * stride_x + gid.y * stride_y + gid.z * stride_z];
    float ym = phi_in[gid.x * stride_x + y_prev * stride_y + gid.z * stride_z];
    float yp = phi_in[gid.x * stride_x + y_next * stride_y + gid.z * stride_z];
    float zm = phi_in[gid.x * stride_x + gid.y * stride_y + z_prev * stride_z];
    float zp = phi_in[gid.x * stride_x + gid.y * stride_y + z_next * stride_z];
    
    float h2 = p.grid_spacing * p.grid_spacing;
    
    // Jacobi iteration: φ_new = (sum of neighbors - h² * 4πGρ) / 6
    phi_out[idx] = (xm + xp + ym + yp + zm + zp - h2 * gravity_4pi * rho[idx]) / 6.0f;
}

// -----------------------------------------------------------------------------
// Kernel: Clear field (set to zero)
// -----------------------------------------------------------------------------

kernel void clear_field(
    device float* field [[buffer(0)]],
    constant uint& num_elements [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_elements) return;
    field[gid] = 0.0f;
}

// -----------------------------------------------------------------------------
// Kernel: Derive temperature field from scattered mass + heat
// -----------------------------------------------------------------------------
// Given:
// - mass_field: scattered mass-per-cell (same units as particle masses)
// - heat_field: scattered heat-per-cell Q_cell (same units as particle heat)
//
// Derive:
//   T_cell = Q_cell / (mass_field * c_v + eps)
//
// This matches the host-side formula previously used in Python:
//   denom = mass_field * max(c_v, eps) + eps
//   T = heat_field / denom
//
kernel void derive_temperature_field(
    device const float* mass_field [[buffer(0)]],
    device const float* heat_field [[buffer(1)]],
    device float* temp_field       [[buffer(2)]],
    constant float& specific_heat  [[buffer(3)]],
    constant uint& num_elements    [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_elements) return;
    // [CHOICE] cell temperature from scattered stores (explicit vacuum semantics)
    // [FORMULA] If m_cell > 0:  T_cell = Q_cell / (m_cell c_v)
    //          Else:           T_cell = 0  (vacuum / empty cell has no temperature)
    // [REASON] removes numerical ε and makes vacuum handling explicit
    // [NOTES] Requires invariant c_v > 0. If c_v <= 0, we write NaN to fail loudly.
    float cv = specific_heat;
    if (!(cv > 0.0f)) {
        temp_field[gid] = qnan_f();
        return;
    }
    float m = mass_field[gid];
    if (m > 0.0f) {
        temp_field[gid] = heat_field[gid] / (m * cv);
    } else {
        temp_field[gid] = 0.0f;
    }
}

// =============================================================================
// Particle-Particle Interaction Kernel (Collision + Excitation Transfer)
// =============================================================================
// This kernel computes short-range forces between particles:
// 1. Soft-sphere repulsion: prevents overlap, stronger for excited particles
// 2. Excitation transfer: when particles "bump", excitation equilibrates
//
// NOTE: This is O(N²) which is fine for N < 1000. For larger systems,
// use spatial hashing or neighbor lists.

struct ParticleInteractionParams {
    uint32_t num_particles;
    float dt;
    float particle_radius;       // r: particle radius for collision detection
    float young_modulus;         // E: Young's modulus for Hertzian contact (spring stiffness)
    float thermal_conductivity;  // k: heat transfer on contact
    float specific_heat;         // c_v: heat capacity per unit mass
    float restitution;           // e: coefficient of restitution (0-1)
};

kernel void particle_interactions(
    device float* particle_pos            [[buffer(0)]],  // N * 3 (read-only for positions)
    device float* particle_vel            [[buffer(1)]],  // N * 3 (read-write for velocity)
    device float* particle_excitation     [[buffer(2)]],  // N (read-write for excitation)
    device const float* particle_mass     [[buffer(3)]],  // N (read-only)
    device float* particle_heat           [[buffer(4)]],  // N (read-write for heat)
    device const float* particle_vel_in   [[buffer(5)]],  // N * 3 (snapshot for consistent reads)
    device const float* particle_heat_in  [[buffer(6)]],  // N (snapshot for consistent reads)
    constant ParticleInteractionParams& p [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.num_particles) return;
    
    // Read this particle's state
    float3 pos_i = float3(
        particle_pos[gid * 3 + 0],
        particle_pos[gid * 3 + 1],
        particle_pos[gid * 3 + 2]
    );
    float3 vel_i = float3(
        particle_vel[gid * 3 + 0],
        particle_vel[gid * 3 + 1],
        particle_vel[gid * 3 + 2]
    );
    float mass_i = particle_mass[gid];
    float heat_i = particle_heat[gid];
    // Note: particle_excitation is read-only intrinsic property, not needed for collisions

    // [CHOICE] collision kernel invariants (fail loudly)
    // [FORMULA] require: m_i>0, c_v>0, r>0, dt>0
    // [REASON] silent clamps hide invalid physical states
    // [NOTES] on violation we write NaNs to outputs for this particle.
    if (!(mass_i > 0.0f) || !(p.specific_heat > 0.0f) || !(p.particle_radius > 0.0f) || !(p.dt > 0.0f)) {
        float qn = qnan_f();
        particle_vel[gid * 3 + 0] = qn;
        particle_vel[gid * 3 + 1] = qn;
        particle_vel[gid * 3 + 2] = qn;
        particle_heat[gid] = qn;
        return;
    }
    
    // Particle radius (from material property)
    float r_i = p.particle_radius;
    
    // Accumulate impulses and heat changes
    float3 impulse_total = float3(0.0f);
    float heat_delta = 0.0f;
    
    // Loop over all other particles
    for (uint j = 0; j < p.num_particles; j++) {
        if (j == gid) continue;
        
        float3 pos_j = float3(
            particle_pos[j * 3 + 0],
            particle_pos[j * 3 + 1],
            particle_pos[j * 3 + 2]
        );
        float3 vel_j = float3(
            particle_vel_in[j * 3 + 0],
            particle_vel_in[j * 3 + 1],
            particle_vel_in[j * 3 + 2]
        );
        float mass_j = particle_mass[j];
        float heat_j = particle_heat_in[j];

        if (!(mass_j > 0.0f)) {
            float qn = qnan_f();
            particle_vel[gid * 3 + 0] = qn;
            particle_vel[gid * 3 + 1] = qn;
            particle_vel[gid * 3 + 2] = qn;
            particle_heat[gid] = qn;
            return;
        }
        
        float r_j = p.particle_radius;
        float combined_radius = r_i + r_j;
        
        // Distance vector from j to i
        float3 delta = pos_i - pos_j;
        float dist = length(delta);
        
        if (dist < combined_radius && dist > 1e-6f) {
            // =====================================================
            // COLLISION DETECTED
            // =====================================================
            float3 n = delta / dist;  // Normal from j to i
            float overlap = combined_radius - dist;
            
            // Relative velocity (i relative to j)
            float3 v_rel = vel_i - vel_j;
            float v_n = dot(v_rel, n);  // Normal component
            
            // -------------------------------------------------
            // IMPULSE-BASED COLLISION (momentum conservation)
            // -------------------------------------------------
            if (v_n < 0.0f) {  // Only if approaching
                // Coefficient of restitution e: v'_rel = -e * v_rel
                float e = p.restitution;
                
                // Reduced mass: m_eff = m_i * m_j / (m_i + m_j)
                float m_eff = (mass_i * mass_j) / (mass_i + mass_j);
                
                // Impulse magnitude: J = (1 + e) * m_eff * |v_n|
                float J = (1.0f + e) * m_eff * (-v_n);
                
                // Impulse on particle i: Δv_i = J/m_i * n
                // We divide by 2 because we process each pair twice (once for i, once for j)
                impulse_total += (n * J / mass_i) * 0.5f;
                
                // -------------------------------------------------
                // ENERGY CONSERVATION: KE_lost becomes heat
                // -------------------------------------------------
                // KE_before = 0.5 * m_eff * v_n^2
                // KE_after  = 0.5 * m_eff * (e * v_n)^2
                // ΔKE = 0.5 * m_eff * v_n^2 * (1 - e^2)
                float ke_lost = 0.5f * m_eff * v_n * v_n * (1.0f - e * e);
                heat_delta += ke_lost * 0.5f;  // Half to each particle
            }
            
            // -------------------------------------------------
            // HERTZIAN CONTACT FORCE (prevents interpenetration)
            // -------------------------------------------------
            // F = (4/3) * E* * sqrt(R*) * δ^(3/2) for Hertzian contact
            // Simplified: F ≈ E * δ for small overlaps (linear spring)
            float contact_force = p.young_modulus * overlap;
            impulse_total += n * contact_force * p.dt / mass_i;
            
            // -------------------------------------------------
            // HEAT CONDUCTION ON CONTACT (Fourier's law)
            // -------------------------------------------------
            // Q = k * A * (T_j - T_i) / d, where d ≈ overlap
            // Approximate: dQ/dt ∝ k * (T_j - T_i) * contact_area
            // Temperature consistency: T = Q / (m * c_v)
            // [CHOICE] contact conduction temperature mapping
            // [FORMULA] T = Q / (m c_v)
            // [REASON] consistent with particle internal energy definition
            // [NOTES] invariants m>0, c_v>0 enforced above (no silent eps clamps).
            float cv = p.specific_heat;
            float T_i = heat_i / (mass_i * cv);
            float T_j = heat_j / (mass_j * cv);
            float contact_area = overlap * overlap;  // Approximate circular contact
            float dQ_conduction = p.thermal_conductivity * contact_area * (T_j - T_i) * p.dt;
            heat_delta += dQ_conduction;
            
            // Note: Excitation (oscillator frequency) is an INTRINSIC property
            // and does NOT equilibrate on contact. Each particle maintains its
            // unique frequency throughout the simulation.
        }
    }
    
    // Apply accumulated changes
    vel_i += impulse_total;
    heat_i += heat_delta;
    
    // [CHOICE] non-negative thermal energy (0 K baseline)
    // [FORMULA] Q >= 0
    // [REASON] internal thermal energy relative to absolute zero cannot be negative
    heat_i = max(heat_i, 0.0f);
    
    // Write back
    particle_vel[gid * 3 + 0] = vel_i.x;
    particle_vel[gid * 3 + 1] = vel_i.y;
    particle_vel[gid * 3 + 2] = vel_i.z;
    // Note: particle_excitation is NOT written - it's an intrinsic property
    particle_heat[gid] = heat_i;
}

// =============================================================================
// SPATIAL HASH GRID ACCELERATION
// =============================================================================
// Three-phase approach for O(N) collision detection:
//   Phase 1: Assign each particle to a cell (compute cell index)
//   Phase 2: Count particles per cell, compute prefix sum → cell start indices
//   Phase 3: Collision detection using cell-based neighbor lookup
//
// This reduces O(N²) to O(N * k) where k = avg particles in 27 neighbor cells.
// For uniform distributions, k ~ 27 * (N / num_cells) which is constant for
// fixed density, giving O(N) total complexity.
// =============================================================================

// -----------------------------------------------------------------------------
// Utility: Compute cell index from position
// -----------------------------------------------------------------------------
inline uint3 position_to_cell(
    float3 pos,
    float inv_cell_size,
    float3 domain_min,
    uint3 grid_dims
) {
    // [CHOICE] periodic spatial hash domain
    // [FORMULA] cell = floor(((pos-domain_min)/h) mod grid_dims)
    // [REASON] collision neighborhood should match torus/periodic simulation domain
    float3 g = (pos - domain_min) * inv_cell_size;
    float3 gd = float3(grid_dims);
    g = g - gd * floor(g / gd); // wrap into [0,grid_dims)
    return uint3(floor(g));
}

inline uint cell_to_linear(uint3 cell, uint3 grid_dims) {
    return cell.x * grid_dims.y * grid_dims.z + cell.y * grid_dims.z + cell.z;
}

inline uint3 linear_to_cell(uint linear_idx, uint3 grid_dims) {
    uint x = linear_idx / (grid_dims.y * grid_dims.z);
    uint rem = linear_idx % (grid_dims.y * grid_dims.z);
    uint y = rem / grid_dims.z;
    uint z = rem % grid_dims.z;
    return uint3(x, y, z);
}

// -----------------------------------------------------------------------------
// Kernel: Assign particles to cells (Phase 1)
// -----------------------------------------------------------------------------
// Each particle computes its cell index and stores it.
// Also atomically increments the cell's particle count.

kernel void spatial_hash_assign(
    device const float* particle_pos       [[buffer(0)]],  // N * 3
    device uint* particle_cell_idx         [[buffer(1)]],  // N (output: linear cell index)
    device atomic_uint* cell_counts        [[buffer(2)]],  // num_cells (output: count per cell)
    constant SpatialHashParams& p          [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.num_particles) return;
    
    float3 pos = float3(
        particle_pos[gid * 3 + 0],
        particle_pos[gid * 3 + 1],
        particle_pos[gid * 3 + 2]
    );
    
    float3 domain_min = float3(p.domain_min_x, p.domain_min_y, p.domain_min_z);
    uint3 grid_dims = uint3(p.grid_x, p.grid_y, p.grid_z);
    
    uint3 cell = position_to_cell(pos, p.inv_cell_size, domain_min, grid_dims);
    uint linear_idx = cell_to_linear(cell, grid_dims);
    
    particle_cell_idx[gid] = linear_idx;
    atomic_fetch_add_explicit(&cell_counts[linear_idx], 1, memory_order_relaxed);
}

// -----------------------------------------------------------------------------
// Kernel: Exclusive prefix sum on cell counts (Phase 2a)
// -----------------------------------------------------------------------------
// Computes cell_starts[i] = sum(cell_counts[0..i-1])
// This gives the starting index in the sorted particle array for each cell.
//
// For small grids (< 64³ = 262k cells), single-thread scan is acceptable.
// For larger grids, use parallel Blelloch scan.

kernel void spatial_hash_prefix_sum(
    device const uint* cell_counts         [[buffer(0)]],  // num_cells
    device uint* cell_starts               [[buffer(1)]],  // num_cells + 1
    constant uint& num_cells               [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Single-thread sequential scan (for num_cells up to ~256k)
    if (gid != 0) return;
    
    uint running_sum = 0;
    for (uint i = 0; i < num_cells; i++) {
        cell_starts[i] = running_sum;
        running_sum += cell_counts[i];
    }
    cell_starts[num_cells] = running_sum;  // Total particle count
}

// Parallel Blelloch-style prefix sum for larger grids
// This uses threadgroup-local reductions for better scaling
kernel void spatial_hash_prefix_sum_parallel(
    device uint* cell_counts               [[buffer(0)]],  // num_cells (in/out: becomes cell_starts)
    device uint* block_sums                [[buffer(1)]],  // (num_cells / BLOCK_SIZE) intermediate sums
    constant uint& num_cells               [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup uint* shared [[threadgroup(0)]]
) {
    // Load into shared memory
    uint idx = gid;
    shared[tid] = (idx < num_cells) ? cell_counts[idx] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Up-sweep (reduce) phase
    for (uint stride = 1; stride < tg_size; stride *= 2) {
        uint ai = (tid + 1) * stride * 2 - 1;
        if (ai < tg_size) {
            shared[ai] += shared[ai - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store block sum and clear last element
    if (tid == tg_size - 1) {
        uint block_idx = gid / tg_size;
        block_sums[block_idx] = shared[tid];
        shared[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Down-sweep phase
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        uint ai = (tid + 1) * stride * 2 - 1;
        if (ai < tg_size) {
            uint t = shared[ai - stride];
            shared[ai - stride] = shared[ai];
            shared[ai] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write back exclusive prefix sum
    if (idx < num_cells) {
        cell_counts[idx] = shared[tid];
    }
}

// -----------------------------------------------------------------------------
// Generic kernel: u32 exclusive scan (parallel, block-hierarchical)
// -----------------------------------------------------------------------------
// [CHOICE] parallel prefix sum (exclusive) for uint32 buffers
// [FORMULA] out[i] = Σ_{j < i} in[j]
// [REASON] required for O(N) spatial hash and spectral frequency binning without CPU sync
// [NOTES] This is implemented as a block scan + hierarchical scan of block sums.
//
// Pass 1: per-block exclusive scan, emitting `block_sums[block]`.
kernel void exclusive_scan_u32_pass1(
    device const uint* in                [[buffer(0)]],  // n
    device uint* out                     [[buffer(1)]],  // n
    device uint* block_sums              [[buffer(2)]],  // num_blocks
    constant uint& n                     [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup uint* shared             [[threadgroup(0)]]
) {
    uint idx = tg_id * tg_size + tid;
    shared[tid] = (idx < n) ? in[idx] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep
    for (uint stride = 1; stride < tg_size; stride <<= 1) {
        uint ai = ((tid + 1u) * stride * 2u) - 1u;
        if (ai < tg_size) shared[ai] += shared[ai - stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint total = shared[tg_size - 1u];
    if (tid == tg_size - 1u) {
        block_sums[tg_id] = total;
        shared[tg_size - 1u] = 0u; // exclusive
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        uint ai = ((tid + 1u) * stride * 2u) - 1u;
        if (ai < tg_size) {
            uint t = shared[ai - stride];
            shared[ai - stride] = shared[ai];
            shared[ai] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (idx < n) out[idx] = shared[tid];
}

// Pass 2/3 helper: add scanned block offsets to per-block scan output.
kernel void exclusive_scan_u32_add_block_offsets(
    device uint* out                      [[buffer(0)]],  // n (in/out)
    device const uint* block_prefix       [[buffer(1)]],  // num_blocks (exclusive scan of block_sums)
    constant uint& n                      [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint idx = tg_id * tg_size + tid;
    if (idx >= n) return;
    out[idx] += block_prefix[tg_id];
}

// Optional helper: write total sum as out[n] for (n+1)-length start arrays.
kernel void exclusive_scan_u32_finalize_total(
    device const uint* in                 [[buffer(0)]],  // n
    device uint* out                      [[buffer(1)]],  // n+1 (first n already filled with exclusive scan)
    constant uint& n                      [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    if (n == 0) { out[0] = 0u; return; }
    out[n] = out[n - 1u] + in[n - 1u];
}

// -----------------------------------------------------------------------------
// Kernel: Scatter particles to sorted array (Phase 2b)
// -----------------------------------------------------------------------------
// Places particle indices into a sorted array based on their cell.
// Uses atomic counters per cell to handle collisions within cells.

kernel void spatial_hash_scatter(
    device const uint* particle_cell_idx   [[buffer(0)]],  // N (cell index per particle)
    device uint* sorted_particle_idx       [[buffer(1)]],  // N (output: sorted indices)
    device atomic_uint* cell_offsets       [[buffer(2)]],  // num_cells (working offsets)
    constant uint& num_particles           [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_particles) return;
    
    uint cell_idx = particle_cell_idx[gid];
    uint slot = atomic_fetch_add_explicit(&cell_offsets[cell_idx], 1, memory_order_relaxed);
    sorted_particle_idx[slot] = gid;
}

// =============================================================================
// Spectral carrier frequency binning (GPU-only)
// =============================================================================
// Buckets carriers by ω_k to enable sparse coupling by scanning only nearby bins.
//
// This is designed to be exact w.r.t. fp32 tuning semantics:
// we will later choose bin width such that bins beyond a fixed neighborhood
// contribute exactly 0 to the Gaussian in fp32 (exp-underflow).

// -----------------------------------------------------------------------------
// Constants / types for spectral binning (must appear before kernels)
// -----------------------------------------------------------------------------
// [CHOICE] fp32 exp underflow boundary for Gaussian tuning
// [FORMULA] Let FLT_TRUE_MIN = 2^-149. exp(-x) underflows to 0 in fp32 for x >= x0,
//           where x0 = -ln(FLT_TRUE_MIN) = -ln(2^-149) = 149 * ln(2).
// [REASON] enables exact sparsity: interactions with (Δω^2/σ^2) >= x0 contribute
//          exactly 0 in fp32, so they are provably irrelevant.
constant float kFp32ExpUnderflowX0 = 103.27893f; // 149*ln(2) (rounded to fp32)

struct SpectralBinParams {
    float omega_min;
    float inv_bin_width;
};

// [CHOICE] float→ordered-u32 mapping for atomic min/max
// [FORMULA] key = (sign? ~u : (u ^ 0x80000000)), where u is IEEE-754 bits of float
// [REASON] enables atomic_min/atomic_max on floats using atomic_uint while preserving ordering
inline uint float_to_ordered_u32(float x) {
    uint u = as_type<uint>(x);
    uint sign = u & 0x80000000u;
    return (sign != 0u) ? ~u : (u ^ 0x80000000u);
}

inline float ordered_u32_to_float(uint key) {
    uint sign = key & 0x80000000u;
    uint u = (sign != 0u) ? (key ^ 0x80000000u) : ~key;
    return as_type<float>(u);
}

// Prototypes (definitions appear later in file).
inline void atomic_max_uint_device(device atomic_uint* address, uint val);
inline void atomic_min_uint_device(device atomic_uint* address, uint val);

kernel void spectral_reduce_omega_minmax_keys(
    device const float* carrier_omega       [[buffer(0)]],  // maxM
    device const uint* num_carriers_in      [[buffer(1)]],  // (1,) snapshot
    device atomic_uint* omega_min_key       [[buffer(2)]],  // (1,) init = 0xFFFFFFFF
    device atomic_uint* omega_max_key       [[buffer(3)]],  // (1,) init = 0
    uint gid [[thread_position_in_grid]]
) {
    uint n = (num_carriers_in != nullptr) ? num_carriers_in[0] : 0u;
    if (gid >= n) return;
    float w = carrier_omega[gid];
    if (!isfinite(w)) return;
    uint key = float_to_ordered_u32(w);
    atomic_min_uint_device(&omega_min_key[0], key);
    atomic_max_uint_device(&omega_max_key[0], key);
}

kernel void spectral_compute_bin_params(
    device const atomic_uint* omega_min_key [[buffer(0)]], // (1,)
    device const atomic_uint* omega_max_key [[buffer(1)]], // (1,)
    device const uint* num_carriers_in      [[buffer(2)]], // (1,)
    device SpectralBinParams* out_params    [[buffer(3)]], // (1,)
    constant float& gate_width_max          [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    uint n = (num_carriers_in != nullptr) ? max(num_carriers_in[0], 1u) : 1u;

    float wmin = ordered_u32_to_float(atomic_load_explicit(&omega_min_key[0], memory_order_relaxed));
    float wmax = ordered_u32_to_float(atomic_load_explicit(&omega_max_key[0], memory_order_relaxed));
    float range = wmax - wmin;

    // [CHOICE] fp32-exact coupling support radius for Gaussian
    // [FORMULA] R_max = sqrt(x0) * σ_max, where x0 = -ln(FLT_TRUE_MIN)
    // [REASON] outside this radius, exp(-(Δω/σ)^2) underflows to 0 in fp32 (exactly)
    float R_max = sqrt(kFp32ExpUnderflowX0) * gate_width_max;

    // [CHOICE] bin width (derived, no knob)
    // [FORMULA] W = max(R_max, range / n)
    // [REASON] ensures finite binning resolution without user-tunable parameters
    float W = max(R_max, (n > 0u) ? (range / (float)n) : R_max);
    if (!(W > 0.0f)) {
        out_params[0].omega_min = qnan_f();
        out_params[0].inv_bin_width = qnan_f();
        return;
    }

    out_params[0].omega_min = wmin;
    out_params[0].inv_bin_width = 1.0f / W;
}

kernel void spectral_bin_count_carriers(
    device const float* carrier_omega       [[buffer(0)]],  // maxM
    device const uint* num_carriers_in      [[buffer(1)]],  // (1,)
    device atomic_uint* bin_counts          [[buffer(2)]],  // num_bins
    device const SpectralBinParams* bin_p   [[buffer(3)]],  // (1,)
    constant uint& num_bins                 [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = (num_carriers_in != nullptr) ? num_carriers_in[0] : 0u;
    if (gid >= n) return;
    float w = carrier_omega[gid];
    float f = (w - bin_p[0].omega_min) * bin_p[0].inv_bin_width;
    int bi = (int)floor(f);
    bi = clamp(bi, 0, (int)num_bins - 1);
    atomic_fetch_add_explicit(&bin_counts[(uint)bi], 1u, memory_order_relaxed);
}

kernel void spectral_bin_scatter_carriers(
    device const float* carrier_omega       [[buffer(0)]],  // maxM
    device const uint* num_carriers_in      [[buffer(1)]],  // (1,)
    device atomic_uint* bin_offsets         [[buffer(2)]],  // num_bins (working copy of starts)
    device const SpectralBinParams* bin_p   [[buffer(3)]],  // (1,)
    constant uint& num_bins                 [[buffer(4)]],
    device uint* carrier_binned_idx         [[buffer(5)]],  // maxM
    uint gid [[thread_position_in_grid]]
) {
    uint n = (num_carriers_in != nullptr) ? num_carriers_in[0] : 0u;
    if (gid >= n) return;
    float w = carrier_omega[gid];
    float f = (w - bin_p[0].omega_min) * bin_p[0].inv_bin_width;
    int bi = (int)floor(f);
    bi = clamp(bi, 0, (int)num_bins - 1);
    uint slot = atomic_fetch_add_explicit(&bin_offsets[(uint)bi], 1u, memory_order_relaxed);
    carrier_binned_idx[slot] = gid;
}

// -----------------------------------------------------------------------------
// Kernel: Spatial hash collision detection (Phase 3)
// -----------------------------------------------------------------------------
// For each particle, check only particles in the same cell and 26 neighbors.
// This is O(N * k) where k = avg particles per 27-cell neighborhood.

kernel void spatial_hash_collisions(
    // Particle state
    device const float* particle_pos       [[buffer(0)]],  // N * 3
    device float* particle_vel             [[buffer(1)]],  // N * 3
    device float* particle_excitation      [[buffer(2)]],  // N
    device const float* particle_mass      [[buffer(3)]],  // N
    device float* particle_heat            [[buffer(4)]],  // N
    // Spatial hash data
    device const uint* sorted_particle_idx [[buffer(5)]],  // N (sorted by cell)
    device const uint* cell_starts         [[buffer(6)]],  // num_cells + 1
    device const uint* particle_cell_idx   [[buffer(7)]],  // N (cell index per particle)
    // Snapshot inputs for consistent reads (avoid write hazards)
    device const float* particle_vel_in    [[buffer(8)]],  // N * 3
    device const float* particle_heat_in   [[buffer(9)]],  // N
    // Parameters
    constant SpatialCollisionParams& p     [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.num_particles) return;
    
    // Read this particle's state
    float3 pos_i = float3(
        particle_pos[gid * 3 + 0],
        particle_pos[gid * 3 + 1],
        particle_pos[gid * 3 + 2]
    );
    float3 vel_i = float3(
        particle_vel[gid * 3 + 0],
        particle_vel[gid * 3 + 1],
        particle_vel[gid * 3 + 2]
    );
    float mass_i = particle_mass[gid];
    float heat_i = particle_heat[gid];
    // Note: particle_excitation is read-only intrinsic property, not needed for collisions

    // Collision kernel invariants (fail loudly). See particle_interactions for rationale.
    if (!(mass_i > 0.0f) || !(p.specific_heat > 0.0f) || !(p.particle_radius > 0.0f) || !(p.dt > 0.0f)) {
        float qn = qnan_f();
        particle_vel[gid * 3 + 0] = qn;
        particle_vel[gid * 3 + 1] = qn;
        particle_vel[gid * 3 + 2] = qn;
        particle_heat[gid] = qn;
        return;
    }
    
    float r_i = p.particle_radius;
    uint3 grid_dims = uint3(p.grid_x, p.grid_y, p.grid_z);
    
    // Get this particle's cell
    float3 domain_min = float3(p.domain_min_x, p.domain_min_y, p.domain_min_z);
    uint3 cell_i = position_to_cell(pos_i, p.inv_cell_size, domain_min, grid_dims);
    
    // Accumulate impulses and changes
    float3 impulse_total = float3(0.0f);
    float heat_delta = 0.0f;
    
    // Iterate over 3x3x3 neighborhood (27 cells)
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                // [CHOICE] periodic neighbor wrap
                // [FORMULA] neighbor = (cell_i + d) mod grid_dims
                // [REASON] consistent with periodic domain; no boundary “dead zones”
                int3 neighbor = int3(cell_i) + int3(dx, dy, dz);
                neighbor.x = (neighbor.x % (int)p.grid_x + (int)p.grid_x) % (int)p.grid_x;
                neighbor.y = (neighbor.y % (int)p.grid_y + (int)p.grid_y) % (int)p.grid_y;
                neighbor.z = (neighbor.z % (int)p.grid_z + (int)p.grid_z) % (int)p.grid_z;

                uint neighbor_linear = cell_to_linear(uint3(neighbor), grid_dims);
                uint start = cell_starts[neighbor_linear];
                uint end = cell_starts[neighbor_linear + 1];
                
                // Iterate over particles in this cell
                for (uint slot = start; slot < end; slot++) {
                    uint j = sorted_particle_idx[slot];
                    if (j == gid) continue;  // Skip self
                    
                    float3 pos_j = float3(
                        particle_pos[j * 3 + 0],
                        particle_pos[j * 3 + 1],
                        particle_pos[j * 3 + 2]
                    );
                    
                    float3 delta = pos_i - pos_j;
                    float dist_sq = dot(delta, delta);
                    float r_j = p.particle_radius;
                    float combined_radius = r_i + r_j;
                    
                    // Early exit with squared distance check (avoid sqrt)
                    if (dist_sq >= combined_radius * combined_radius || dist_sq < 1e-12f) {
                        continue;
                    }
                    
                    float dist = sqrt(dist_sq);
                    
                    // =====================================================
                    // COLLISION DETECTED
                    // =====================================================
                    float3 n = delta / dist;
                    float overlap = combined_radius - dist;
                    
                    float3 vel_j = float3(
                        particle_vel_in[j * 3 + 0],
                        particle_vel_in[j * 3 + 1],
                        particle_vel_in[j * 3 + 2]
                    );
                    float mass_j = particle_mass[j];
                    float heat_j = particle_heat_in[j];

                    if (!(mass_j > 0.0f)) {
                        float qn = qnan_f();
                        particle_vel[gid * 3 + 0] = qn;
                        particle_vel[gid * 3 + 1] = qn;
                        particle_vel[gid * 3 + 2] = qn;
                        particle_heat[gid] = qn;
                        return;
                    }
                    
                    float3 v_rel = vel_i - vel_j;
                    float v_n = dot(v_rel, n);
                    
                    // IMPULSE-BASED COLLISION
                    if (v_n < 0.0f) {
                        float e = p.restitution;
                        float m_eff = (mass_i * mass_j) / (mass_i + mass_j);
                        float J = (1.0f + e) * m_eff * (-v_n);
                        impulse_total += (n * J / mass_i) * 0.5f;
                        
                        // Energy conservation
                        float ke_lost = 0.5f * m_eff * v_n * v_n * (1.0f - e * e);
                        heat_delta += ke_lost * 0.5f;
                    }
                    
                    // HERTZIAN CONTACT FORCE
                    float contact_force = p.young_modulus * overlap;
                    impulse_total += n * contact_force * p.dt / mass_i;
                    
                    // HEAT CONDUCTION
                    // Temperature consistency: T = Q / (m * c_v)
                    float cv = p.specific_heat;
                    float T_i = heat_i / (mass_i * cv);
                    float T_j = heat_j / (mass_j * cv);
                    float contact_area = overlap * overlap;
                    float dQ_conduction = p.thermal_conductivity * contact_area * (T_j - T_i) * p.dt;
                    heat_delta += dQ_conduction;
                    
                    // Note: Excitation (oscillator frequency) is INTRINSIC - no equilibration
                }
            }
        }
    }
    
    // Apply accumulated changes
    vel_i += impulse_total;
    heat_i += heat_delta;
    
    // Physical constraints
    heat_i = max(heat_i, 0.0f);
    
    // Write back
    particle_vel[gid * 3 + 0] = vel_i.x;
    particle_vel[gid * 3 + 1] = vel_i.y;
    particle_vel[gid * 3 + 2] = vel_i.z;
    // Note: particle_excitation NOT written - intrinsic property
    particle_heat[gid] = heat_i;
}

// -----------------------------------------------------------------------------
// Helper: Atomic Float Add for Threadgroup Memory (CAS Loop)
// -----------------------------------------------------------------------------
// Metal doesn't have native atomic float in threadgroup memory on all hardware.
// We emulate it using compare-and-swap on uint, interpreting the bits as float.

inline void atomic_add_float_threadgroup(threadgroup atomic_uint* address, float val) {
    uint old_val = atomic_load_explicit(address, memory_order_relaxed);
    uint new_val;
    
    while (true) {
        // Convert bits to float to do the math
        float old_f = as_type<float>(old_val);
        float new_f = old_f + val;
        
        // Convert result back to bits
        new_val = as_type<uint>(new_f);
        
        // Try to swap: if *address == old_val, set *address = new_val and return true
        // If *address != old_val, update old_val to current value and return false
        if (atomic_compare_exchange_weak_explicit(
                address, 
                &old_val, 
                new_val, 
                memory_order_relaxed, 
                memory_order_relaxed)) {
            break;
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel: Tiled scatter with threadgroup hash accumulation (exact)
// -----------------------------------------------------------------------------
// Goal: reduce global atomic contention of particle→grid deposition.
//
// Strategy:
// - Each threadgroup processes a contiguous tile of particles.
// - Deposits are accumulated into a fixed-size threadgroup hash table keyed by
//   linear cell index. Values are float accumulators stored as atomic_uint bits.
// - At the end, each occupied hash slot flushes exactly once to global atomics.
// - If the local hash table cannot place a key within kMaxProbe probes, we fall
//   back to direct global atomic adds for that deposit (still exact).
//
// Correctness: identical to `scatter_particle` (same trilinear weights), no
// approximation; only a different accumulation order (floating point assoc).
//
constant uint kScatterHashSize = 2048u; // power of two
constant uint kScatterEmptyKey = 0xFFFFFFFFu;
constant uint kScatterMaxProbe = 32u;

inline uint scatter_hash_u32(uint x) {
    return hash_u32(x ^ 0xB5297A4Du);
}

kernel void scatter_particle_tiled(
    device const float* particle_pos      [[buffer(0)]],  // N * 3
    device const float* particle_mass     [[buffer(1)]],  // N
    device const float* particle_heat     [[buffer(2)]],  // N
    device const float* particle_energy   [[buffer(3)]],  // N
    device atomic_float* gravity_field    [[buffer(4)]],  // X * Y * Z
    device atomic_float* heat_field       [[buffer(5)]],  // X * Y * Z
    constant TiledScatterParams& p        [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    threadgroup atomic_uint* tg_keys      [[threadgroup(0)]], // kScatterHashSize
    threadgroup atomic_uint* tg_g         [[threadgroup(1)]], // kScatterHashSize (float bits)
    threadgroup atomic_uint* tg_h         [[threadgroup(2)]]  // kScatterHashSize (float bits)
) {
    // 1) Initialize hash table
    for (uint i = tid; i < kScatterHashSize; i += tg_size) {
        atomic_store_explicit(&tg_keys[i], kScatterEmptyKey, memory_order_relaxed);
        atomic_store_explicit(&tg_g[i], 0u, memory_order_relaxed);
        atomic_store_explicit(&tg_h[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) Process one particle per thread (tile_size == threads_per_threadgroup)
    if (gid < p.num_particles) {
        float3 pos = float3(
            particle_pos[gid * 3 + 0],
            particle_pos[gid * 3 + 1],
            particle_pos[gid * 3 + 2]
        );
        float mass = particle_mass[gid];
        float heat = particle_heat[gid];
        float e_osc = particle_energy[gid];
        float e_total = heat + e_osc;

        // Trilinear coords (periodic)
        uint3 base_idx;
        float3 frac;
        uint3 grid_dims = uint3(p.grid_x, p.grid_y, p.grid_z);
        trilinear_coords(pos, p.inv_grid_spacing, grid_dims, base_idx, frac);

        float wx0 = 1.0f - frac.x, wx1 = frac.x;
        float wy0 = 1.0f - frac.y, wy1 = frac.y;
        float wz0 = 1.0f - frac.z, wz1 = frac.z;

        float weights[8] = {
            wx0 * wy0 * wz0,  // 000
            wx0 * wy0 * wz1,  // 001
            wx0 * wy1 * wz0,  // 010
            wx0 * wy1 * wz1,  // 011
            wx1 * wy0 * wz0,  // 100
            wx1 * wy0 * wz1,  // 101
            wx1 * wy1 * wz0,  // 110
            wx1 * wy1 * wz1   // 111
        };

        uint gx = p.grid_x, gy = p.grid_y, gz = p.grid_z;
        uint x0 = base_idx.x;
        uint y0 = base_idx.y;
        uint z0 = base_idx.z;
        uint x1 = (x0 + 1) % gx;
        uint y1 = (y0 + 1) % gy;
        uint z1 = (z0 + 1) % gz;

        uint stride_z = 1;
        uint stride_y = gz;
        uint stride_x = gy * gz;
        auto idx3 = [&](uint x, uint y, uint z) -> uint {
            return x * stride_x + y * stride_y + z * stride_z;
        };

        uint idxs[8] = {
            idx3(x0, y0, z0),
            idx3(x0, y0, z1),
            idx3(x0, y1, z0),
            idx3(x0, y1, z1),
            idx3(x1, y0, z0),
            idx3(x1, y0, z1),
            idx3(x1, y1, z0),
            idx3(x1, y1, z1),
        };

        // Deposit each corner into threadgroup hash
        for (uint c = 0; c < 8; c++) {
            uint key = idxs[c];
            float w = weights[c];
            float add_g = mass * w;
            float add_h = e_total * w;

            uint h0 = scatter_hash_u32(key) & (kScatterHashSize - 1u);
            bool placed = false;
            for (uint probe = 0; probe < kScatterMaxProbe; probe++) {
                uint slot = (h0 + probe) & (kScatterHashSize - 1u);
                uint old = atomic_load_explicit(&tg_keys[slot], memory_order_relaxed);
                if (old == key) {
                    atomic_add_float_threadgroup(&tg_g[slot], add_g);
                    atomic_add_float_threadgroup(&tg_h[slot], add_h);
                    placed = true;
                    break;
                }
                if (old == kScatterEmptyKey) {
                    uint expected = kScatterEmptyKey;
                    if (atomic_compare_exchange_weak_explicit(
                            &tg_keys[slot], &expected, key,
                            memory_order_relaxed, memory_order_relaxed)) {
                        atomic_add_float_threadgroup(&tg_g[slot], add_g);
                        atomic_add_float_threadgroup(&tg_h[slot], add_h);
                        placed = true;
                        break;
                    }
                }
            }
            if (!placed) {
                // Fallback: exact global atomic adds for this deposit.
                atomic_fetch_add_explicit(&gravity_field[key], add_g, memory_order_relaxed);
                atomic_fetch_add_explicit(&heat_field[key], add_h, memory_order_relaxed);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) Flush hash table to global atomics
    for (uint i = tid; i < kScatterHashSize; i += tg_size) {
        uint key = atomic_load_explicit(&tg_keys[i], memory_order_relaxed);
        if (key == kScatterEmptyKey) continue;

        uint gbits = atomic_load_explicit(&tg_g[i], memory_order_relaxed);
        uint hbits = atomic_load_explicit(&tg_h[i], memory_order_relaxed);
        float gsum = as_type<float>(gbits);
        float hsum = as_type<float>(hbits);

        if (gsum != 0.0f) atomic_fetch_add_explicit(&gravity_field[key], gsum, memory_order_relaxed);
        if (hsum != 0.0f) atomic_fetch_add_explicit(&heat_field[key], hsum, memory_order_relaxed);
    }
}

// -----------------------------------------------------------------------------
// Helper: Atomic Max for Threadgroup/Device Memory (CAS Loop)
// -----------------------------------------------------------------------------
// Replaces atomic_max_explicit which may not be supported for all types/spaces.

inline void atomic_max_uint_threadgroup(threadgroup atomic_uint* address, uint val) {
    uint old_val = atomic_load_explicit(address, memory_order_relaxed);
    while (val > old_val) {
        if (atomic_compare_exchange_weak_explicit(
                address, &old_val, val, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

inline void atomic_max_uint_device(device atomic_uint* address, uint val) {
    uint old_val = atomic_load_explicit(address, memory_order_relaxed);
    while (val > old_val) {
        if (atomic_compare_exchange_weak_explicit(
                address, &old_val, val, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

inline void atomic_min_uint_device(device atomic_uint* address, uint val) {
    uint old_val = atomic_load_explicit(address, memory_order_relaxed);
    while (val < old_val) {
        if (atomic_compare_exchange_weak_explicit(
                address, &old_val, val, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

// -----------------------------------------------------------------------------
// Spectral Carrier Coupling (Resonance Potential, Langevin Flow)
// -----------------------------------------------------------------------------
// This implements a conservative "resonance potential" view of the spectral layer.
//
// Definitions:
// - Oscillator (particle in wave-space): z_i = A_i e^{iθ_i}
// - Carrier (global mode):              C_k = R_k e^{iψ_k}
//
// Potential (conceptual):
//   U = - Σ_{i,k} T_{ik}(ω_i, ω_k, σ_k) * Re(z_i C_k*)
//       + (λ/2) Σ_k |C_k|^2
//
// where T_{ik} is a Gaussian tuning kernel in frequency space.
//
// Gradients:
// - Carrier "force":   ∂(-U)/∂C_k*  = Σ_i T_{ik} z_i  - λ C_k
// - Phase "torque":    θ̇_i += Σ_k T_{ik} (A_i R_k) sin(ψ_k - θ_i)
//
// Langevin flow:
// - Add isotropic noise with temperature T to both carrier updates and phase updates.

// -----------------------------------------------------------------------------
// Carrier memory (anchored + crystallized)
// -----------------------------------------------------------------------------
// We model "chunks" as long-lived spectral modes (carriers) that store a small
// set of anchored oscillators and their relative phase offsets.
//
// This yields:
// - Storage: crystallized carriers stop decaying and stop drifting in ω.
// - Top-down bias: crystallized carriers pull anchored oscillators toward stored
//   phase offsets and can inject energy into anchored oscillators.
// - Idle compute: same kernels with a mode knob (consolidate/disambiguate/explore).
//
#define CARRIER_ANCHORS 8u
#define CARRIER_STATE_VOLATILE 0u
#define CARRIER_STATE_STABLE 1u
#define CARRIER_STATE_CRYSTALLIZED 2u

struct SpectralCarrierParams {
    uint32_t num_osc;              // N
    uint32_t max_carriers;         // capacity of carrier arrays
    uint32_t num_carriers;         // current active carriers (<= max_carriers)
    float dt;
    float coupling_scale;          // phase torque scale
    float carrier_reg;             // λ (L2 regularization on |C| to prevent blow-up)
    uint32_t rng_seed;             // updated each tick by host
    float conflict_threshold;      // coherence threshold to trigger split (high = stricter)
    float offender_weight_floor;   // ignore tiny weights
    float gate_width_min;
    float gate_width_max;
    float ema_alpha;               // smoothing for conflict
    float recenter_alpha;          // smoothing for ω_k recentering
    // --- Memory / modes ---
    uint32_t mode;                 // 0=online, 1=consolidate, 2=disambiguate, 3=explore
    float anchor_random_eps;       // ε-greedy anchor refresh probability
    float stable_amp_threshold;    // promote volatile->stable when |C| exceeds this
    float crystallize_amp_threshold;       // stable->crystallized when |C| exceeds this...
    float crystallize_conflict_threshold;  // ...and conflict below this for long enough
    uint32_t crystallize_age;      // consecutive stable frames required
    float crystallized_coupling_boost;     // extra coupling for crystallized carriers
    float volatile_decay_mul;      // extra decay factor for volatile carriers
    float stable_decay_mul;        // extra decay factor for stable carriers
    float crystallized_decay_mul;  // extra decay factor for crystallized carriers
    float topdown_phase_scale;     // extra phase pull for anchored oscillators
    float topdown_energy_scale;    // energy injection scale for crystallized carriers
    float topdown_random_energy_eps; // random energy nudge probability (exploration)
    float repulsion_scale;         // carrier ω repulsion (disambiguation)
};

inline float tuning_from_freq(float omega_i, float omega_k, float gate_width) {
    float d = omega_i - omega_k;
    // [CHOICE] tuning kernel width (invariant)
    // [FORMULA] σ = gate_width, requires σ>0
    // [REASON] removes eps clamp; invalid σ should fail loudly upstream
    // [NOTES] host enforces gate_width_min>0; kernel state clamps gate_width into [min,max].
    if (!(gate_width > 0.0f)) return qnan_f();
    float sigma = gate_width;
    return exp(-(d * d) / (sigma * sigma));
}

// -----------------------------------------------------------------------------
// Kernel: Parallel Force Accumulation (Oscillator-Centric)
// -----------------------------------------------------------------------------
// Replaces the O(N) loop inside carrier threads with an O(N) parallel kernel.
// Each oscillator computes its contribution to ALL carriers.
// Uses threadgroup memory to reduce global atomic contention.

struct CarrierAccumulators {
    atomic_float force_r;
    atomic_float force_i;
    atomic_float w_sum;
    atomic_float w_omega_sum;
    atomic_float w_omega2_sum;
    atomic_float w_amp_sum;
    atomic_uint offender_score; // float bits
    atomic_uint offender_idx;   // uint
};

kernel void spectral_accumulate_forces(
    // Oscillator state
    device const float* osc_phase           [[buffer(0)]],  // N
    device const float* osc_omega           [[buffer(1)]],  // N
    device const float* osc_amp             [[buffer(2)]],  // N
    // Carrier state (read-only)
    device const float* carrier_omega       [[buffer(3)]],  // maxM
    device const float* carrier_gate_width  [[buffer(4)]],  // maxM
    device const float* carrier_conflict    [[buffer(5)]],  // maxM
    // Output accumulators
    device CarrierAccumulators* accums      [[buffer(6)]],  // maxM
    // Parameters
    constant SpectralCarrierParams& p       [[buffer(7)]],
    device const uint* num_carriers_in      [[buffer(8)]], // (1,) uint32/int32
    // Sparse binning inputs
    device const uint* bin_starts           [[buffer(9)]],  // num_bins + 1
    device const uint* carrier_binned_idx   [[buffer(10)]], // maxM (indices in [0,num_carriers))
    device const SpectralBinParams* bin_p   [[buffer(11)]], // (1,) {omega_min, inv_bin_width}
    constant uint& num_bins                 [[buffer(12)]],
    uint gid [[thread_position_in_grid]]
) {
    uint num_carriers = (num_carriers_in != nullptr) ? num_carriers_in[0] : 0u;
    num_carriers = min(num_carriers, p.max_carriers);
    if (gid >= p.num_osc) return;
    if (num_carriers == 0u) return;
    if (!(num_bins > 0u)) return;

    float omega_i = osc_omega[gid];
    float amp_i = osc_amp[gid];
    float phi_i = osc_phase[gid];

    // Precompute z_i
    float zr = amp_i * cos(phi_i);
    float zi = amp_i * sin(phi_i);

    // [CHOICE] bin neighborhood radius
    // [FORMULA] radius = 2 bins guarantees covering |Δω|<=R_max when bin_width>=R_max
    // [REASON] includes boundary cases where tuning_from_freq() is still nonzero at |Δω|≈R_max
    const int rad = 2;
    float fbin = (omega_i - bin_p[0].omega_min) * bin_p[0].inv_bin_width;
    int bin_i = (int)floor(fbin);
    int b0 = clamp(bin_i - rad, 0, (int)num_bins - 1);
    int b1 = clamp(bin_i + rad, 0, (int)num_bins - 1);

    for (int b = b0; b <= b1; b++) {
        uint start = bin_starts[(uint)b];
        uint end = bin_starts[(uint)b + 1u];
        for (uint j = start; j < end; j++) {
            uint k = carrier_binned_idx[j];
            if (k >= num_carriers) continue;

            float omega_k = carrier_omega[k];
            float gate_w = clamp(carrier_gate_width[k], p.gate_width_min, p.gate_width_max);
            float t = tuning_from_freq(omega_i, omega_k, gate_w);
            float w = t * amp_i;
            if (w <= p.offender_weight_floor) continue;

            device CarrierAccumulators& g_acc = accums[k];
            atomic_fetch_add_explicit(&g_acc.force_r, w * zr, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_acc.force_i, w * zi, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_acc.w_sum, w, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_acc.w_omega_sum, w * omega_i, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_acc.w_omega2_sum, w * omega_i * omega_i, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_acc.w_amp_sum, w * amp_i, memory_order_relaxed);

            if (carrier_conflict[k] > p.conflict_threshold) {
                uint score_bits = as_type<uint>(w);
                atomic_max_uint_device(&g_acc.offender_score, score_bits);
                // Best-effort index update (race-tolerant).
                if (atomic_load_explicit(&g_acc.offender_score, memory_order_relaxed) == score_bits) {
                    atomic_store_explicit(&g_acc.offender_idx, gid, memory_order_relaxed);
                }
            }
        }
    }
}

kernel void spectral_carrier_update_and_split(
    // Oscillator state (particles-as-oscillators)
    device const float* osc_phase           [[buffer(0)]],  // N
    device const float* osc_omega           [[buffer(1)]],  // N
    device const float* osc_amp             [[buffer(2)]],  // N
    // Carrier state (in/out)
    device float* carrier_real              [[buffer(3)]],  // maxM
    device float* carrier_imag              [[buffer(4)]],  // maxM
    device float* carrier_omega             [[buffer(5)]],  // maxM
    device float* carrier_gate_width        [[buffer(6)]],  // maxM
    device float* carrier_conflict          [[buffer(7)]],  // maxM (out: spectral variance proxy)
    // Memory state (in/out)
    device uint* carrier_state              [[buffer(8)]],  // maxM (0=volatile,1=stable,2=crystallized)
    device uint* carrier_age                [[buffer(9)]],  // maxM (consecutive stable frames)
    device uint* carrier_anchor_idx         [[buffer(10)]], // maxM * CARRIER_ANCHORS (osc indices, UINT_MAX=empty)
    device float* carrier_anchor_phase      [[buffer(11)]], // maxM * CARRIER_ANCHORS (phase offsets)
    device float* carrier_anchor_weight     [[buffer(12)]], // maxM * CARRIER_ANCHORS (anchor strengths)
    // Split output
    device atomic_uint* out_num_carriers    [[buffer(13)]], // single counter (in/out)
    device uint* out_spawned_from_osc       [[buffer(14)]], // maxM (optional: osc index for spawned carrier)
    // Random phases for spawned carriers (pre-generated on device)
    device const float* random_phases       [[buffer(15)]], // maxM (uniform [0,1])
    // Global system energy statistics (computed via reduction pass)
    device const float* energy_stats        [[buffer(16)]], // (4,) [mean_abs, mean, std, count]
    constant SpectralCarrierParams& p       [[buffer(17)]],
    // NEW: Pre-accumulated forces
    device CarrierAccumulators* accums      [[buffer(18)]],
    device const uint* num_carriers_in      [[buffer(19)]], // (1,) uint32/int32 snapshot
    uint gid [[thread_position_in_grid]]
) {
    // One thread per *existing* carrier (snapshot count, stable during this pass).
    uint current = (num_carriers_in != nullptr) ? num_carriers_in[0] : 0u;
    current = min(current, p.max_carriers);
    if (gid >= current) return;

    float cr = carrier_real[gid];
    float ci = carrier_imag[gid];
    float omega_k = carrier_omega[gid];
    float gate_w = clamp(carrier_gate_width[gid], p.gate_width_min, p.gate_width_max);
    uint state = carrier_state[gid];
    uint age = carrier_age[gid];

    // -------------------------------------------------------------------------
    // Adaptive renormalization (global, thermodynamic)
    // -------------------------------------------------------------------------
    float mean_abs_e = energy_stats[0];
    // [CHOICE] global energy scale (explicit zero-energy semantics)
    // [FORMULA] e_scale = mean_abs_e; if e_scale==0 -> no adaptive decay/no noise
    // [REASON] removes eps clamp; defines behavior when the system has zero energy
    float e_scale = mean_abs_e;
    float adaptive_decay = 1.0f;
    if (e_scale > 0.0f) {
        adaptive_decay = exp(-p.dt / e_scale);
    }
    float decay_mul = 1.0f;
    if (state == CARRIER_STATE_VOLATILE) decay_mul = max(p.volatile_decay_mul, 0.0f);
    else if (state == CARRIER_STATE_STABLE) decay_mul = max(p.stable_decay_mul, 0.0f);
    else decay_mul = max(p.crystallized_decay_mul, 0.0f);
    cr *= (adaptive_decay * decay_mul);
    ci *= (adaptive_decay * decay_mul);

    // Read accumulated values from global memory
    device CarrierAccumulators& acc = accums[gid];
    float force_r_raw = atomic_load_explicit(&acc.force_r, memory_order_relaxed);
    float force_i_raw = atomic_load_explicit(&acc.force_i, memory_order_relaxed);
    float w_sum = atomic_load_explicit(&acc.w_sum, memory_order_relaxed);
    float w_omega_sum = atomic_load_explicit(&acc.w_omega_sum, memory_order_relaxed);
    float w_omega2_sum = atomic_load_explicit(&acc.w_omega2_sum, memory_order_relaxed);
    float w_amp_sum = atomic_load_explicit(&acc.w_amp_sum, memory_order_relaxed);
    
    // Offender info
    float offender_score = as_type<float>(atomic_load_explicit(&acc.offender_score, memory_order_relaxed));
    uint offender_idx = atomic_load_explicit(&acc.offender_idx, memory_order_relaxed);

    float mean_omega = omega_k;
    if (w_sum > 0.0f) {
        mean_omega = (w_omega_sum / w_sum);
    }

    // Coherence / conflict:
    float R = sqrt(force_r_raw * force_r_raw + force_i_raw * force_i_raw);
    float coherence = 0.0f;
    if (w_amp_sum > 0.0f) {
        coherence = clamp(R / w_amp_sum, 0.0f, 1.0f);
    }
    float inst_conflict = 1.0f - coherence;

    // Persistent conflict (EMA)
    float prev_conflict = carrier_conflict[gid];
    float a = clamp(p.ema_alpha, 0.0f, 1.0f);
    float conflict = prev_conflict * (1.0f - a) + inst_conflict * a;
    carrier_conflict[gid] = conflict;

    // Re-center ω_k
    if (state != CARRIER_STATE_CRYSTALLIZED) {
        float rc = clamp(p.recenter_alpha, 0.0f, 1.0f);
        omega_k = omega_k * (1.0f - rc) + mean_omega * rc;
    }

    // [CHOICE] adaptive gate width (frequency spread of current supporters)
    // [FORMULA] σ_k^2 = E_w[ω^2] - (E_w[ω])^2, where E_w uses weights w=t*A_i
    // [REASON] restores “open/sharpen” behavior as an emergent property:
    //          - broad supporter band ⇒ larger σ (recruit/generalize)
    //          - narrow supporter band ⇒ smaller σ (specialize/focus)
    // [NOTES] - This is not driven by coherence/conflict; those drive splitting.
    //         - For crystallized carriers, σ is frozen (like ω_k) to preserve memory identity.
    if (state != CARRIER_STATE_CRYSTALLIZED && w_sum > 0.0f) {
        float Ew = w_omega_sum / w_sum;
        float Ew2 = w_omega2_sum / w_sum;
        float var = Ew2 - (Ew * Ew);
        var = max(var, 0.0f); // numerical floor (variance is non-negative)
        float sigma_hat = sqrt(var);
        float rc = clamp(p.recenter_alpha, 0.0f, 1.0f);
        gate_w = gate_w * (1.0f - rc) + sigma_hat * rc;
    }
    gate_w = clamp(gate_w, p.gate_width_min, p.gate_width_max);

    // Mean-field normalization
    float force_r = 0.0f;
    float force_i = 0.0f;
    if (w_sum > 0.0f) {
        float inv_w = 1.0f / w_sum;
        force_r = force_r_raw * inv_w;
        force_i = force_i_raw * inv_w;
    }

    // Metabolic homeostasis
    float income = sqrt(force_r * force_r + force_i * force_i);
    float expense = e_scale;
    if (state != CARRIER_STATE_CRYSTALLIZED && expense > 0.0f && income < expense) {
        float deficit = expense - income;
        float shrink = exp(-p.dt * deficit / expense);
        cr *= shrink;
        ci *= shrink;
    }

    // Langevin carrier update
    float2 n = randn2(p.rng_seed ^ 0xA5A5A5A5u, gid);
    float std_e = energy_stats[2];
    float disorder = 0.0f;
    if (e_scale > 0.0f) {
        disorder = clamp(std_e / e_scale, 0.0f, 10.0f);
    }
    float noise_scale = sqrt(max(2.0f * (disorder * e_scale) * p.dt, 0.0f));
    float reg = (state == CARRIER_STATE_CRYSTALLIZED) ? 0.0f : p.carrier_reg;
    float dcr = (force_r - reg * cr) * p.dt + noise_scale * n.x;
    float dci = (force_i - reg * ci) * p.dt + noise_scale * n.y;
    cr += dcr;
    ci += dci;

    // Repulsion (idle mode)
    if (p.mode == 2u && p.repulsion_scale > 0.0f && current > 1u && state != CARRIER_STATE_CRYSTALLIZED) {
        float repel = 0.0f;
        for (uint k2 = 0; k2 < current; k2++) {
            if (k2 == gid) continue;
            float d = omega_k - carrier_omega[k2];
            float s = gate_w + clamp(carrier_gate_width[k2], p.gate_width_min, p.gate_width_max);
            // gate widths are invariants (>0); if violated, fail loudly.
            if (!(s > 0.0f)) {
                carrier_omega[gid] = qnan_f();
                return;
            }
            repel += d * exp(-(d * d) / (s * s));
        }
        omega_k += p.dt * p.repulsion_scale * repel;
    }

    // Crystallization state machine
    float ampC = sqrt(cr * cr + ci * ci);
    if (state == CARRIER_STATE_VOLATILE) {
        if (ampC >= max(p.stable_amp_threshold, 0.0f)) {
            state = CARRIER_STATE_STABLE;
            age = 0u;
        }
    }
    if (state == CARRIER_STATE_STABLE) {
        bool ok_amp = ampC >= max(p.crystallize_amp_threshold, 0.0f);
        bool ok_conf = conflict <= clamp(p.crystallize_conflict_threshold, 0.0f, 1.0f);
        if (ok_amp && ok_conf) {
            age += 1u;
            if (age >= max(p.crystallize_age, 1u)) {
                state = CARRIER_STATE_CRYSTALLIZED;
                age = p.crystallize_age;
            }
        } else {
            age = 0u;
        }
    }

    // Anchor refresh
    if (state != CARRIER_STATE_CRYSTALLIZED && p.num_osc > 0u) {
        uint h = hash_u32(p.rng_seed ^ (gid * 0xB4B82E39u) ^ 0x1C3A5F7Du);
        float u = u01_from_u32(h);
        uint slot = hash_u32(h + 0x3C6EF372u) % CARRIER_ANCHORS;
        uint base = gid * CARRIER_ANCHORS + slot;
        float eps_anchor = clamp(p.anchor_random_eps, 0.0f, 1.0f);
        
        if (u <= eps_anchor) {
            uint cand = hash_u32(h + 0x9E3779B9u) % p.num_osc;
            carrier_anchor_idx[base] = cand;
            float psi = atan2(ci, cr);
            float d = osc_phase[cand] - psi;
            d = d - 2.0f * M_PI_F * floor((d + M_PI_F) / (2.0f * M_PI_F));
            carrier_anchor_phase[base] = d;
            float w = tuning_from_freq(osc_omega[cand], omega_k, gate_w) * osc_amp[cand];
            carrier_anchor_weight[base] = w;
        } else if (offender_score > 0.0f) {
            carrier_anchor_idx[base] = offender_idx;
            float psi = atan2(ci, cr);
            float d = osc_phase[offender_idx] - psi;
            d = d - 2.0f * M_PI_F * floor((d + M_PI_F) / (2.0f * M_PI_F));
            carrier_anchor_phase[base] = d;
            float w = tuning_from_freq(osc_omega[offender_idx], omega_k, gate_w) * osc_amp[offender_idx];
            carrier_anchor_weight[base] = w;
        }
    }

    carrier_real[gid] = cr;
    carrier_imag[gid] = ci;
    carrier_omega[gid] = omega_k;
    carrier_gate_width[gid] = gate_w;
    carrier_state[gid] = state;
    carrier_age[gid] = age;

    // Conflict-driven split
    if (state != CARRIER_STATE_CRYSTALLIZED && offender_score > 0.0f && conflict > p.conflict_threshold) {
        uint slot = atomic_fetch_add_explicit(out_num_carriers, 1, memory_order_relaxed);
        if (slot < p.max_carriers) {
            float omega_new = osc_omega[offender_idx];
            float amp_new = osc_amp[offender_idx];
            float phi_new = osc_phase[offender_idx];
            float init_scale = 0.5f;
            float nr = init_scale * amp_new * cos(phi_new);
            float ni = init_scale * amp_new * sin(phi_new);

            carrier_real[slot] = nr;
            carrier_imag[slot] = ni;
            carrier_omega[slot] = omega_new;
            carrier_gate_width[slot] = gate_w;
            carrier_conflict[slot] = 0.0f;
            carrier_state[slot] = CARRIER_STATE_VOLATILE;
            carrier_age[slot] = 0u;

            for (uint j = 0; j < CARRIER_ANCHORS; j++) {
                uint b = slot * CARRIER_ANCHORS + j;
                carrier_anchor_idx[b] = 0xFFFFFFFFu;
                carrier_anchor_phase[b] = 0.0f;
                carrier_anchor_weight[b] = 0.0f;
            }
            {
                uint b0 = slot * CARRIER_ANCHORS;
                carrier_anchor_idx[b0] = offender_idx;
                float psi0 = atan2(ni, nr);
                float d0 = phi_new - psi0;
                d0 = d0 - 2.0f * M_PI_F * floor((d0 + M_PI_F) / (2.0f * M_PI_F));
                carrier_anchor_phase[b0] = d0;
                carrier_anchor_weight[b0] = amp_new;
            }

            if (out_spawned_from_osc != nullptr) {
                out_spawned_from_osc[slot] = offender_idx;
            }

            float r = random_phases[slot] * 2.0f * M_PI_F;
            float rot_r = cos(r);
            float rot_i = sin(r);
            float rr = carrier_real[slot];
            float ri = carrier_imag[slot];
            carrier_real[slot] = rr * rot_r - ri * rot_i;
            carrier_imag[slot] = rr * rot_i + ri * rot_r;
        }
        carrier_conflict[gid] = 0.0f;
    }
}

kernel void spectral_update_oscillator_phases(
    device float* osc_phase               [[buffer(0)]],  // N (in/out)
    device const float* osc_omega         [[buffer(1)]],  // N
    device const float* osc_amp           [[buffer(2)]],  // N
    device const float* carrier_real      [[buffer(3)]],  // maxM
    device const float* carrier_imag      [[buffer(4)]],  // maxM
    device const float* carrier_omega     [[buffer(5)]],  // maxM
    device const float* carrier_gate_width[[buffer(6)]],  // maxM
    device const uint* carrier_state      [[buffer(7)]],  // maxM
    device const uint* carrier_anchor_idx [[buffer(8)]],  // maxM * CARRIER_ANCHORS
    device const float* carrier_anchor_phase [[buffer(9)]], // maxM * CARRIER_ANCHORS
    device const float* carrier_anchor_weight[[buffer(10)]], // maxM * CARRIER_ANCHORS
    // Global energy stats (same format as in carrier update)
    device const float* energy_stats      [[buffer(11)]],  // (4,) [mean_abs, mean, std, count]
    device const uint* num_carriers_in    [[buffer(12)]], // (1,) uint32/int32 snapshot
    constant SpectralCarrierParams& p     [[buffer(13)]],
    // Sparse binning inputs
    device const uint* bin_starts         [[buffer(14)]],  // num_bins + 1
    device const uint* carrier_binned_idx [[buffer(15)]],  // maxM
    device const SpectralBinParams* bin_p [[buffer(16)]],  // (1,)
    constant uint& num_bins               [[buffer(17)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.num_osc) return;
    uint num_carriers = (num_carriers_in != nullptr) ? num_carriers_in[0] : 0u;
    num_carriers = min(num_carriers, p.max_carriers);

    float phi = osc_phase[gid];
    float omega_i = osc_omega[gid];
    float amp_i = osc_amp[gid];

    // Torque from resonance potential:
    //   θ̇_i += Σ_k T_ik (A_i R_k) sin(ψ_k - θ_i)
    float torque = 0.0f;
    const int rad = 2;
    if (num_carriers > 0u && num_bins > 0u) {
        float fbin = (omega_i - bin_p[0].omega_min) * bin_p[0].inv_bin_width;
        int bin_i = (int)floor(fbin);
        int b0 = clamp(bin_i - rad, 0, (int)num_bins - 1);
        int b1 = clamp(bin_i + rad, 0, (int)num_bins - 1);
        for (int b = b0; b <= b1; b++) {
            uint start = bin_starts[(uint)b];
            uint end = bin_starts[(uint)b + 1u];
            for (uint jj = start; jj < end; jj++) {
                uint k = carrier_binned_idx[jj];
                if (k >= num_carriers) continue;

                float omega_k = carrier_omega[k];
                float gate_w = clamp(carrier_gate_width[k], p.gate_width_min, p.gate_width_max);
                float t = tuning_from_freq(omega_i, omega_k, gate_w);
                float cr = carrier_real[k];
                float ci = carrier_imag[k];
                float psi = atan2(ci, cr);
                float R = sqrt(cr * cr + ci * ci);
                float boost = 1.0f;
                if (carrier_state[k] == CARRIER_STATE_CRYSTALLIZED) {
                    boost += max(p.crystallized_coupling_boost, 0.0f);
                }
                torque += boost * t * (amp_i * R) * sin(psi - phi);

                // Extra top-down phase pull if this oscillator is anchored in a crystallized carrier.
                if (carrier_state[k] == CARRIER_STATE_CRYSTALLIZED && p.topdown_phase_scale > 0.0f) {
                    uint base = k * CARRIER_ANCHORS;
                    for (uint j = 0; j < CARRIER_ANCHORS; j++) {
                        uint idx = carrier_anchor_idx[base + j];
                        if (idx == gid) {
                            float off = carrier_anchor_phase[base + j];
                            float w = carrier_anchor_weight[base + j];
                            float target = psi + off;
                            float d = target - phi;
                            d = d - 2.0f * M_PI_F * floor((d + M_PI_F) / (2.0f * M_PI_F));
                            torque += p.topdown_phase_scale * w * sin(d);
                        }
                    }
                }
            }
        }
    }

    float mean_abs_e = energy_stats[0];
    float std_e = energy_stats[2];
    // [CHOICE] global energy scale (explicit zero-energy semantics)
    // [FORMULA] e_scale = mean_abs_e; if e_scale==0 -> disorder=noise=0
    // [REASON] removes eps clamp; defines behavior when energy statistics vanish
    float e_scale = mean_abs_e;
    float disorder = 0.0f;
    if (e_scale > 0.0f) {
        disorder = clamp(std_e / e_scale, 0.0f, 10.0f);
    }
    float noise_scale = sqrt(max(2.0f * (disorder * e_scale) * p.dt, 0.0f));
    float n = randn1(p.rng_seed ^ 0xC3C3C3C3u, gid);
    float dphi = omega_i + p.coupling_scale * torque;
    phi += dphi * p.dt + noise_scale * n;

    // Wrap phase to [0, 2π)
    phi = phi - 2.0f * M_PI_F * floor(phi / (2.0f * M_PI_F));
    osc_phase[gid] = phi;
}

// -----------------------------------------------------------------------------
// Kernel: Top-down energy bias from crystallized carriers (anchored completion)
// -----------------------------------------------------------------------------
// One thread per carrier. Atomically injects small energy into anchored oscillators
// proportional to carrier activation and anchor weights.
kernel void spectral_topdown_bias_energies(
    device atomic_float* osc_energy        [[buffer(0)]],  // N (in/out, atomic adds)
    device const float* osc_amp            [[buffer(1)]],  // N
    device const uint* carrier_state       [[buffer(2)]],  // maxM
    device const uint* carrier_anchor_idx  [[buffer(3)]],  // maxM * CARRIER_ANCHORS
    device const float* carrier_anchor_weight [[buffer(4)]], // maxM * CARRIER_ANCHORS
    device const uint* num_carriers_in     [[buffer(5)]], // (1,) uint32/int32 snapshot
    constant SpectralCarrierParams& p      [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint num_carriers = (num_carriers_in != nullptr) ? num_carriers_in[0] : 0u;
    num_carriers = min(num_carriers, p.max_carriers);
    if (gid >= num_carriers) return;
    if (carrier_state[gid] != CARRIER_STATE_CRYSTALLIZED) return;
    if (p.topdown_energy_scale <= 0.0f || p.num_osc == 0u) return;

    uint base = gid * CARRIER_ANCHORS;
    float wsum = 0.0f;
    float act = 0.0f;
    for (uint j = 0; j < CARRIER_ANCHORS; j++) {
        uint idx = carrier_anchor_idx[base + j];
        if (idx == 0xFFFFFFFFu || idx >= p.num_osc) continue;
        float w = carrier_anchor_weight[base + j];
        wsum += w;
        act += w * osc_amp[idx];
    }
    if (!(wsum > 0.0f)) return;
    act = act / wsum;

    // Inject energy into anchored oscillators, favoring low-amplitude ones.
    for (uint j = 0; j < CARRIER_ANCHORS; j++) {
        uint idx = carrier_anchor_idx[base + j];
        if (idx == 0xFFFFFFFFu || idx >= p.num_osc) continue;
        float w = carrier_anchor_weight[base + j] / wsum;
        float a = osc_amp[idx];
        float need = 1.0f / (1.0f + a); // higher when oscillator is quiet
        float dE = p.dt * p.topdown_energy_scale * act * w * need;
        if (dE != 0.0f) {
            atomic_fetch_add_explicit(&osc_energy[idx], dE, memory_order_relaxed);
        }
    }

    // Exploration: random nudge so weaker bonds can "get lucky".
    if (p.topdown_random_energy_eps > 0.0f) {
        uint h = hash_u32(p.rng_seed ^ (gid * 0x27D4EB2Du) ^ 0x85EBCA6Bu);
        float u = u01_from_u32(h);
        if (u <= clamp(p.topdown_random_energy_eps, 0.0f, 1.0f)) {
            uint idx = hash_u32(h + 0x165667B1u) % p.num_osc;
            float dE = p.dt * (0.25f * p.topdown_energy_scale) * act;
            atomic_fetch_add_explicit(&osc_energy[idx], dE, memory_order_relaxed);
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel: Spawn carriers for uncoupled oscillators
// -----------------------------------------------------------------------------
// Any oscillator with total coupling weight below threshold spawns its own carrier.
// This ensures every oscillator is coupled to at least one carrier.

kernel void spectral_spawn_uncoupled(
    device const float* osc_phase           [[buffer(0)]],  // N
    device const float* osc_omega           [[buffer(1)]],  // N
    device const float* osc_amp             [[buffer(2)]],  // N
    device const float* carrier_omega       [[buffer(3)]],  // maxM
    device const float* carrier_gate_width  [[buffer(4)]],  // maxM
    device float* carrier_real              [[buffer(5)]],  // maxM (out for new carriers)
    device float* carrier_imag              [[buffer(6)]],  // maxM
    device float* carrier_omega_out         [[buffer(7)]],  // maxM
    device float* carrier_gate_width_out    [[buffer(8)]],  // maxM
    device float* carrier_conflict          [[buffer(9)]],  // maxM
    device uint* carrier_state              [[buffer(10)]], // maxM
    device uint* carrier_age                [[buffer(11)]], // maxM
    device uint* carrier_anchor_idx         [[buffer(12)]], // maxM * CARRIER_ANCHORS
    device float* carrier_anchor_phase      [[buffer(13)]], // maxM * CARRIER_ANCHORS
    device float* carrier_anchor_weight     [[buffer(14)]], // maxM * CARRIER_ANCHORS
    device atomic_uint* num_carriers_atomic [[buffer(15)]], // single counter (in/out)
    device const uint* num_carriers_in      [[buffer(16)]], // (1,) uint32/int32 snapshot
    constant uint& max_carriers             [[buffer(17)]],
    constant float& coupling_threshold      [[buffer(18)]], // min total coupling to be "coupled"
    constant float& gate_width_init         [[buffer(19)]],
    constant float& gate_width_min          [[buffer(20)]],
    constant float& gate_width_max          [[buffer(21)]],
    // Sparse binning inputs
    device const uint* bin_starts           [[buffer(22)]], // num_bins + 1
    device const uint* carrier_binned_idx   [[buffer(23)]], // maxM
    device const SpectralBinParams* bin_p   [[buffer(24)]], // (1,)
    constant uint& num_bins                 [[buffer(25)]],
    uint gid [[thread_position_in_grid]],
    constant uint& num_osc                  [[buffer(26)]]
) {
    if (gid >= num_osc) return;
    
    float omega_i = osc_omega[gid];
    float amp_i = osc_amp[gid];
    float phi_i = osc_phase[gid];
    
    uint num_carriers = (num_carriers_in != nullptr) ? num_carriers_in[0] : 0u;
    num_carriers = min(num_carriers, max_carriers);

    // Compute total coupling weight to all existing carriers
    float total_coupling = 0.0f;
    const int rad = 2;
    if (num_carriers > 0u && num_bins > 0u) {
        float fbin = (omega_i - bin_p[0].omega_min) * bin_p[0].inv_bin_width;
        int bin_i = (int)floor(fbin);
        int b0 = clamp(bin_i - rad, 0, (int)num_bins - 1);
        int b1 = clamp(bin_i + rad, 0, (int)num_bins - 1);
        for (int b = b0; b <= b1; b++) {
            uint start = bin_starts[(uint)b];
            uint end = bin_starts[(uint)b + 1u];
            for (uint jj = start; jj < end; jj++) {
                uint k = carrier_binned_idx[jj];
                if (k >= num_carriers) continue;
                float omega_k = carrier_omega[k];
                float gate_w = clamp(carrier_gate_width[k], gate_width_min, gate_width_max);
                float t = tuning_from_freq(omega_i, omega_k, gate_w);
                total_coupling += t;
            }
        }
    }
    
    // If oscillator is not sufficiently coupled to any carrier, spawn its own
    if (total_coupling < coupling_threshold) {
        uint slot = atomic_fetch_add_explicit(num_carriers_atomic, 1, memory_order_relaxed);
        if (slot < max_carriers) {
            // Initialize carrier from oscillator's phasor
            carrier_real[slot] = amp_i * cos(phi_i);
            carrier_imag[slot] = amp_i * sin(phi_i);
            carrier_omega_out[slot] = omega_i;
            carrier_gate_width_out[slot] = gate_width_init;
            carrier_conflict[slot] = 0.0f;
            carrier_state[slot] = CARRIER_STATE_VOLATILE;
            carrier_age[slot] = 0u;
            for (uint j = 0; j < CARRIER_ANCHORS; j++) {
                uint b = slot * CARRIER_ANCHORS + j;
                carrier_anchor_idx[b] = 0xFFFFFFFFu;
                carrier_anchor_phase[b] = 0.0f;
                carrier_anchor_weight[b] = 0.0f;
            }
            // Self-anchor (helps recruitment)
            {
                uint b0 = slot * CARRIER_ANCHORS;
                carrier_anchor_idx[b0] = gid;
                carrier_anchor_phase[b0] = 0.0f;
                carrier_anchor_weight[b0] = amp_i;
            }
        }
    }
}

// =============================================================================
// Particle Generation Kernels
// =============================================================================
// Move synthetic data generation patterns to GPU for faster file injection.

struct ParticleGenParams {
    uint32_t num_particles;
    float grid_x;
    float grid_y;
    float grid_z;
    float energy_scale;
    uint32_t pattern;  // 0=cluster, 1=line, 2=sphere, 3=random, 4=grid
    float center_x;
    float center_y;
    float center_z;
    float spread;      // Cluster spread or sphere radius
    float dir_x;       // Line direction
    float dir_y;
    float dir_z;
};

// -----------------------------------------------------------------------------
// Kernel: Generate particle positions based on pattern
// -----------------------------------------------------------------------------

kernel void generate_particle_positions(
    device float* positions           [[buffer(0)]],  // N * 3
    device const float* random_vals   [[buffer(1)]],  // N * 3 (pre-generated uniform [0,1])
    constant ParticleGenParams& p     [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.num_particles) return;
    
    float3 center = float3(p.center_x, p.center_y, p.center_z);
    float3 r = float3(
        random_vals[gid * 3 + 0],
        random_vals[gid * 3 + 1],
        random_vals[gid * 3 + 2]
    );
    
    float3 pos;
    
    if (p.pattern == 0) {
        // Cluster: Gaussian around center
        // Convert uniform to Gaussian using Box-Muller (approximate)
        float3 gauss = (r - 0.5f) * 2.0f * 2.0f;  // Rough approximation
        pos = center + gauss * p.spread;
    }
    else if (p.pattern == 1) {
        // Line: along direction from start
        float t = float(gid) / float(p.num_particles) * p.spread;
        float3 dir = float3(p.dir_x, p.dir_y, p.dir_z);
        pos = center + dir * t + (r - 0.5f) * 0.5f;
    }
    else if (p.pattern == 2) {
        // Sphere: points on shell
        float theta = r.x * 2.0f * M_PI_F;
        float phi = acos(2.0f * r.y - 1.0f);
        float x = sin(phi) * cos(theta);
        float y = sin(phi) * sin(theta);
        float z = cos(phi);
        pos = center + float3(x, y, z) * p.spread;
    }
    else if (p.pattern == 4) {
        // Grid: regular lattice
        uint side = uint(pow(float(p.num_particles), 1.0f / 3.0f)) + 1;
        uint ix = gid % side;
        uint iy = (gid / side) % side;
        uint iz = gid / (side * side);
        float spacing = min(p.grid_x, min(p.grid_y, p.grid_z)) * 0.8f / float(side);
        pos = float3(
            2.0f + float(ix) * spacing,
            2.0f + float(iy) * spacing,
            2.0f + float(iz) * spacing
        ) + (r - 0.5f) * 0.3f;
    }
    else {
        // Random
        pos = float3(
            r.x * (p.grid_x - 2.0f) + 1.0f,
            r.y * (p.grid_y - 2.0f) + 1.0f,
            r.z * (p.grid_z - 2.0f) + 1.0f
        );
    }
    
    // Clamp to valid range
    float grid_max = min(p.grid_x, min(p.grid_y, p.grid_z)) - 1.5f;
    pos = clamp(pos, float3(0.5f), float3(grid_max));
    
    positions[gid * 3 + 0] = pos.x;
    positions[gid * 3 + 1] = pos.y;
    positions[gid * 3 + 2] = pos.z;
}

// -----------------------------------------------------------------------------
// Kernel: Initialize particle properties (velocity, energy, etc.)
// -----------------------------------------------------------------------------

kernel void initialize_particle_properties(
    device const float* positions      [[buffer(0)]],  // N * 3
    device float* velocities           [[buffer(1)]],  // N * 3
    device float* energies             [[buffer(2)]],  // N
    device float* heats                [[buffer(3)]],  // N
    device float* excitations          [[buffer(4)]],  // N
    device float* masses               [[buffer(5)]],  // N
    device const float* random_vals    [[buffer(6)]],  // N * 4 (for vel_scale, energy, exc, unused)
    constant ParticleGenParams& p      [[buffer(7)]],
    constant float& center_x           [[buffer(8)]],  // Mean position x
    constant float& center_y           [[buffer(9)]],  // Mean position y  
    constant float& center_z           [[buffer(10)]], // Mean position z
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.num_particles) return;
    
    float3 pos = float3(
        positions[gid * 3 + 0],
        positions[gid * 3 + 1],
        positions[gid * 3 + 2]
    );
    
    float3 center = float3(center_x, center_y, center_z);
    
    // Velocity: toward center with small random component
    float3 vel = (center - pos) * 0.01f;
    vel += (float3(random_vals[gid * 4 + 0], random_vals[gid * 4 + 1], random_vals[gid * 4 + 2]) - 0.5f) * 0.1f;
    
    // Energy: distance-based for cluster/sphere, random otherwise
    float energy;
    if (p.pattern == 0 || p.pattern == 2) {
        float dist = length(pos - center);
        float max_dist = p.spread + 1.0f;
        energy = (1.0f - dist / max_dist) * p.energy_scale + 0.1f;
    } else {
        energy = random_vals[gid * 4 + 3] * p.energy_scale * 0.5f + 0.5f;
    }
    energy = max(energy, 0.1f);
    
    // Heat: starts at zero
    float heat = 0.0f;
    
    // Excitation: small random
    float exc = random_vals[gid * 4 + 2] * 0.1f;
    
    // Mass: proportional to energy
    float mass = energy;
    
    velocities[gid * 3 + 0] = vel.x;
    velocities[gid * 3 + 1] = vel.y;
    velocities[gid * 3 + 2] = vel.z;
    energies[gid] = energy;
    heats[gid] = heat;
    excitations[gid] = exc;
    masses[gid] = mass;
}
