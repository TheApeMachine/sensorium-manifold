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

constexpr sampler trilinear_sampler(
    coord::normalized,           // Use [0,1] normalized coordinates
    address::clamp_to_edge,      // Clamp at boundaries
    filter::linear,              // Trilinear interpolation
    mip_filter::none             // No mipmapping
);

// Alternative sampler for nearest-neighbor (for debugging or exact cell access)
constexpr sampler nearest_sampler(
    coord::normalized,
    address::clamp_to_edge,
    filter::nearest
);

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
        float count = max(scratch[0].w, 1.0f);

        float mean_abs = sum_abs / count;
        float mean = sum / count;
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
    // Map to (0,1): avoid exact 0 which breaks log() in Box-Muller.
    // Use 24-bit mantissa-like range for stability.
    float u = (float)(x & 0x00FFFFFFu) * (1.0f / 16777216.0f);
    return max(u, 1e-7f);
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
    // Convert world position to grid coordinates
    float3 grid_pos = pos * inv_spacing;
    
    // Clamp to valid range (leaving room for +1 neighbor)
    grid_pos = clamp(grid_pos, float3(0.0f), float3(grid_dims) - 1.001f);
    
    // Integer base index and fractional part
    base_idx = uint3(floor(grid_pos));
    frac = grid_pos - float3(base_idx);
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
    
    // Base linear index
    uint base = base_idx.x * stride_x + base_idx.y * stride_y + base_idx.z * stride_z;
    
    // Sample all 8 corners
    float c000 = field[base];
    float c001 = field[base + stride_z];
    float c010 = field[base + stride_y];
    float c011 = field[base + stride_y + stride_z];
    float c100 = field[base + stride_x];
    float c101 = field[base + stride_x + stride_z];
    float c110 = field[base + stride_x + stride_y];
    float c111 = field[base + stride_x + stride_y + stride_z];
    
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
    
    uint base = base_idx.x * stride_x + base_idx.y * stride_y + base_idx.z * stride_z;
    
    float c000 = field[base];
    float c001 = field[base + stride_z];
    float c010 = field[base + stride_y];
    float c011 = field[base + stride_y + stride_z];
    float c100 = field[base + stride_x];
    float c101 = field[base + stride_x + stride_z];
    float c110 = field[base + stride_x + stride_y];
    float c111 = field[base + stride_x + stride_y + stride_z];
    
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

kernel void scatter_particles_to_fields(
    device const float* particle_pos      [[buffer(0)]],  // N * 3
    device const float* particle_mass     [[buffer(1)]],  // N
    device const float* particle_heat     [[buffer(2)]],  // N
    device atomic_float* gravity_field    [[buffer(3)]],  // X * Y * Z
    device atomic_float* heat_field       [[buffer(4)]],  // X * Y * Z
    constant ManifoldFieldParams& params  [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.grid_x * params.grid_y * params.grid_z) return;
    
    // This version: each thread handles one particle
    // For large particle counts, consider tiling or hierarchical approaches
}

// Per-particle scatter (one thread per particle)
kernel void scatter_particle(
    device const float* particle_pos      [[buffer(0)]],  // N * 3
    device const float* particle_mass     [[buffer(1)]],  // N
    device const float* particle_heat     [[buffer(2)]],  // N
    device atomic_float* gravity_field    [[buffer(3)]],  // X * Y * Z
    device atomic_float* heat_field       [[buffer(4)]],  // X * Y * Z
    constant ManifoldPhysicsParams& p     [[buffer(5)]],
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
    
    // Grid strides
    uint stride_z = 1;
    uint stride_y = p.grid_z;
    uint stride_x = p.grid_y * p.grid_z;
    uint base = base_idx.x * stride_x + base_idx.y * stride_y + base_idx.z * stride_z;
    
    // Offsets for 8 corners
    uint offsets[8] = {
        0,
        stride_z,
        stride_y,
        stride_y + stride_z,
        stride_x,
        stride_x + stride_z,
        stride_x + stride_y,
        stride_x + stride_y + stride_z
    };
    
    // Atomic scatter to both fields
    for (int i = 0; i < 8; i++) {
        uint idx = base + offsets[i];
        atomic_fetch_add_explicit(&gravity_field[idx], mass * weights[i], memory_order_relaxed);
        atomic_fetch_add_explicit(&heat_field[idx], heat * weights[i], memory_order_relaxed);
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
    // Particle state (read-write)
    device float* particle_pos            [[buffer(2)]],  // N * 3
    device float* particle_vel            [[buffer(3)]],  // N * 3
    device float* particle_energy         [[buffer(4)]],  // N
    device float* particle_heat           [[buffer(5)]],  // N
    device float* particle_excitation     [[buffer(6)]],  // N
    device const float* particle_mass     [[buffer(7)]],  // N (read-only, doesn't change)
    // Parameters
    constant ManifoldPhysicsParams& p     [[buffer(8)]],
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
    
    // -------------------------------------------------------------------------
    // 2. Gather from fields (trilinear interpolation)
    // -------------------------------------------------------------------------
    uint3 base_idx;
    float3 frac;
    uint3 grid_dims = uint3(p.grid_x, p.grid_y, p.grid_z);
    trilinear_coords(pos, p.inv_grid_spacing, grid_dims, base_idx, frac);
    
    // =========================================================================
    // GRAVITATIONAL FORCE: F = -m * ∇φ where φ is gravitational potential
    // =========================================================================
    float3 gravity_grad = sample_gradient_trilinear(
        gravity_potential, base_idx, frac, grid_dims, p.inv_grid_spacing
    );
    // Newton's law of gravitation: F = -G * m * ∇φ
    float3 gravity_force = -gravity_grad * mass * p.G;
    
    // =========================================================================
    // TEMPERATURE AND PRESSURE
    // =========================================================================
    // Sample local temperature field (this is T from heat diffusion on grid)
    float local_temp = sample_field_trilinear(
        temperature_field, base_idx, frac, grid_dims
    );
    
    // Particle temperature from its internal heat: T = Q / (m * c_v)
    float particle_temp = heat / (max(mass, 1e-6f) * p.specific_heat);
    
    // Pressure gradient from ideal gas law: P = ρ * k_B * T
    // Force per unit mass: a = -(1/ρ) * ∇P = -k_B * ∇T (for uniform composition)
    float3 temp_grad = sample_gradient_trilinear(
        temperature_field, base_idx, frac, grid_dims, p.inv_grid_spacing
    );
    float3 pressure_force = -temp_grad * p.k_B * mass;
    
    // =========================================================================
    // HEAT TRANSFER: Newton's law of cooling + Stefan-Boltzmann radiation
    // =========================================================================
    // Newton's law: dQ/dt = h * A * (T_env - T)
    // For sphere: A = 4πr², and h ~ k/r (thermal conductivity / length scale)
    // Combined: dQ/dt ~ k * r * (T_env - T)
    float r = p.particle_radius;
    float heat_transfer_coef = p.thermal_conductivity * r;
    float dQ_conduction = heat_transfer_coef * (local_temp - particle_temp) * p.dt;
    heat += dQ_conduction;
    
    // Stefan-Boltzmann radiation: P = ε * σ * A * T^4
    // Surface area A = 4πr² (we absorb 4π into σ_SB)
    float surface_area = r * r;
    float T4 = particle_temp * particle_temp * particle_temp * particle_temp;
    float dQ_radiation = p.emissivity * p.sigma_SB * surface_area * T4 * p.dt;
    heat = max(heat - dQ_radiation, 0.0f);
    
    // Update temperature after heat exchange
    particle_temp = heat / (max(mass, 1e-6f) * p.specific_heat);
    
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
    // Energy thermalization happens at a fixed rate (not frequency-dependent):
    float tau_thermalization = 10.0f;  // Slower thermalization time scale
    float dQ_thermalization = (energy / tau_thermalization) * p.dt;
    dQ_thermalization = min(dQ_thermalization, energy);
    energy -= dQ_thermalization;
    heat += dQ_thermalization;
    
    // Physical constraints (numerical safety only)
    energy = max(energy, 0.0f);
    heat = max(heat, 0.0f);
    
    // =========================================================================
    // VISCOUS DRAG: Stokes' law F = -6πηrv
    // =========================================================================
    // For a sphere moving through viscous medium at low Reynolds number
    // γ = 6πηr is the drag coefficient
    float gamma = 6.0f * 3.14159f * p.dynamic_viscosity * r;
    
    // Total force = gravity + pressure (hydrostatic equilibrium when balanced)
    float3 total_force = gravity_force + pressure_force;
    
    // Apply forces: F = ma → a = F/m
    float3 acceleration = total_force / max(mass, 1e-6f);
    
    // Clamp acceleration to prevent explosions
    float acc_mag = length(acceleration);
    float max_acc = 10.0f;  // Maximum acceleration per step
    if (acc_mag > max_acc) {
        acceleration = acceleration * (max_acc / acc_mag);
    }
    
    // Compute kinetic energy before damping (for energy conservation)
    float ke_before = 0.5f * mass * dot(vel, vel);
    
    // Apply drag force as a damping term
    // Use exact exponential for accuracy: v' = v * exp(-gamma * dt)
    float damping_factor = exp(-gamma * p.dt);
    vel = vel * damping_factor + acceleration * p.dt;
    
    // Compute kinetic energy after damping
    float ke_after = 0.5f * mass * dot(vel, vel);
    
    // Lost kinetic energy becomes heat (first law of thermodynamics)
    // dE_total/dt = 0, so dE_kinetic + dQ = 0  →  dQ = -dE_kinetic
    // All dissipated kinetic energy becomes internal energy (heat)
    float ke_lost = max(ke_before - ke_after, 0.0f);
    heat += ke_lost;  // 100% energy conservation
    
    // Clamp velocity magnitude to prevent runaway
    float vel_mag = length(vel);
    float max_vel = 2.0f;  // Maximum velocity (aligned with collision kernel)
    if (vel_mag > max_vel) {
        vel = vel * (max_vel / vel_mag);
    }
    
    // -------------------------------------------------------------------------
    // 5. Position update
    // -------------------------------------------------------------------------
    pos += vel * p.dt;
    
    // Clamp to grid bounds with soft boundary (reflect velocity at walls)
    float3 grid_max = float3(p.grid_x, p.grid_y, p.grid_z) * p.grid_spacing * 0.95f;
    float3 grid_min = float3(0.5f);
    
    // Reflect velocity at boundaries
    if (pos.x < grid_min.x) { pos.x = grid_min.x; vel.x = abs(vel.x) * 0.5f; }
    if (pos.y < grid_min.y) { pos.y = grid_min.y; vel.y = abs(vel.y) * 0.5f; }
    if (pos.z < grid_min.z) { pos.z = grid_min.z; vel.z = abs(vel.z) * 0.5f; }
    if (pos.x > grid_max.x) { pos.x = grid_max.x; vel.x = -abs(vel.x) * 0.5f; }
    if (pos.y > grid_max.y) { pos.y = grid_max.y; vel.y = -abs(vel.y) * 0.5f; }
    if (pos.z > grid_max.z) { pos.z = grid_max.z; vel.z = -abs(vel.z) * 0.5f; }
    
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
    
    // Convert position to normalized texture coordinates [0, 1]
    float3 grid_dims = float3(p.grid_x, p.grid_y, p.grid_z);
    float3 tex_coord = pos * p.inv_grid_spacing / grid_dims;
    
    // Sample temperature field using hardware trilinear interpolation
    // Note: texture.sample() returns float4, we use .x for single-channel
    // (gravity potential is only used for gradient, not the value itself)
    float local_temp = temperature_field.sample(trilinear_sampler, tex_coord).x;
    
    // Compute gradients using finite differences on texture samples
    float3 texel_size = 1.0f / grid_dims;
    float3 gravity_grad = float3(
        gravity_potential.sample(trilinear_sampler, tex_coord + float3(texel_size.x, 0, 0)).x -
        gravity_potential.sample(trilinear_sampler, tex_coord - float3(texel_size.x, 0, 0)).x,
        gravity_potential.sample(trilinear_sampler, tex_coord + float3(0, texel_size.y, 0)).x -
        gravity_potential.sample(trilinear_sampler, tex_coord - float3(0, texel_size.y, 0)).x,
        gravity_potential.sample(trilinear_sampler, tex_coord + float3(0, 0, texel_size.z)).x -
        gravity_potential.sample(trilinear_sampler, tex_coord - float3(0, 0, texel_size.z)).x
    ) * (0.5f * p.inv_grid_spacing);
    
    float3 temp_grad = float3(
        temperature_field.sample(trilinear_sampler, tex_coord + float3(texel_size.x, 0, 0)).x -
        temperature_field.sample(trilinear_sampler, tex_coord - float3(texel_size.x, 0, 0)).x,
        temperature_field.sample(trilinear_sampler, tex_coord + float3(0, texel_size.y, 0)).x -
        temperature_field.sample(trilinear_sampler, tex_coord - float3(0, texel_size.y, 0)).x,
        temperature_field.sample(trilinear_sampler, tex_coord + float3(0, 0, texel_size.z)).x -
        temperature_field.sample(trilinear_sampler, tex_coord - float3(0, 0, texel_size.z)).x
    ) * (0.5f * p.inv_grid_spacing);
    
    // Physics calculations (same as buffer version)
    float3 gravity_force = -gravity_grad * mass * p.G;
    float particle_temp = heat / (max(mass, 1e-6f) * p.specific_heat);
    float3 pressure_force = -temp_grad * p.k_B * mass;
    
    // Heat transfer
    float r = p.particle_radius;
    float heat_transfer_coef = p.thermal_conductivity * r;
    float dQ_conduction = heat_transfer_coef * (local_temp - particle_temp) * p.dt;
    heat += dQ_conduction;
    
    float surface_area = r * r;
    float T4 = particle_temp * particle_temp * particle_temp * particle_temp;
    float dQ_radiation = p.emissivity * p.sigma_SB * surface_area * T4 * p.dt;
    heat = max(heat - dQ_radiation, 0.0f);
    
    particle_temp = heat / (max(mass, 1e-6f) * p.specific_heat);
    
    // Excitation is an intrinsic property - do NOT modify it.
    // See gather_update_particles for documentation.
    
    // Energy thermalization (fixed rate, not frequency-dependent)
    float tau_thermalization = 10.0f;
    float dQ_thermalization = (energy / tau_thermalization) * p.dt;
    dQ_thermalization = min(dQ_thermalization, energy);
    energy -= dQ_thermalization;
    heat += dQ_thermalization;
    
    energy = max(energy, 0.0f);
    heat = max(heat, 0.0f);
    
    // Viscous drag
    float gamma = 6.0f * 3.14159f * p.dynamic_viscosity * r;
    float3 total_force = gravity_force + pressure_force;
    float3 acceleration = total_force / max(mass, 1e-6f);
    
    float acc_mag = length(acceleration);
    float max_acc = 10.0f;
    if (acc_mag > max_acc) {
        acceleration = acceleration * (max_acc / acc_mag);
    }
    
    float ke_before = 0.5f * mass * dot(vel, vel);
    float damping_factor = exp(-gamma * p.dt);
    vel = vel * damping_factor + acceleration * p.dt;
    float ke_after = 0.5f * mass * dot(vel, vel);
    float ke_lost = max(ke_before - ke_after, 0.0f);
    heat += ke_lost;
    
    float vel_mag = length(vel);
    float max_vel = 2.0f;
    if (vel_mag > max_vel) {
        vel = vel * (max_vel / vel_mag);
    }
    
    // Position update
    pos += vel * p.dt;
    
    float3 grid_max = float3(p.grid_x, p.grid_y, p.grid_z) * p.grid_spacing * 0.95f;
    float3 grid_min = float3(0.5f);
    
    if (pos.x < grid_min.x) { pos.x = grid_min.x; vel.x = abs(vel.x) * 0.5f; }
    if (pos.y < grid_min.y) { pos.y = grid_min.y; vel.y = abs(vel.y) * 0.5f; }
    if (pos.z < grid_min.z) { pos.z = grid_min.z; vel.z = abs(vel.z) * 0.5f; }
    if (pos.x > grid_max.x) { pos.x = grid_max.x; vel.x = -abs(vel.x) * 0.5f; }
    if (pos.y > grid_max.y) { pos.y = grid_max.y; vel.y = -abs(vel.y) * 0.5f; }
    if (pos.z > grid_max.z) { pos.z = grid_max.z; vel.z = -abs(vel.z) * 0.5f; }
    
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
    
    // 6-point Laplacian stencil with boundary handling
    float xm = (gid.x > 0) ? temp_in[idx - stride_x] : center;
    float xp = (gid.x < p.grid_x - 1) ? temp_in[idx + stride_x] : center;
    float ym = (gid.y > 0) ? temp_in[idx - stride_y] : center;
    float yp = (gid.y < p.grid_y - 1) ? temp_in[idx + stride_y] : center;
    float zm = (gid.z > 0) ? temp_in[idx - stride_z] : center;
    float zp = (gid.z < p.grid_z - 1) ? temp_in[idx + stride_z] : center;
    
    float laplacian = (xm + xp + ym + yp + zm + zp - 6.0f * center) 
                      * (p.inv_grid_spacing * p.inv_grid_spacing);
    
    temp_out[idx] = center + diffusion_coef * laplacian * dt;
}

// -----------------------------------------------------------------------------
// Kernel: Solve Poisson equation for gravity (Jacobi iteration step)
// -----------------------------------------------------------------------------
// ∇²φ = 4πG ρ  →  One Jacobi iteration step
// For production, consider FFT-based solver or multigrid

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
    
    // Boundary handling (Dirichlet: φ = 0 at boundary)
    float xm = (gid.x > 0) ? phi_in[idx - stride_x] : 0.0f;
    float xp = (gid.x < p.grid_x - 1) ? phi_in[idx + stride_x] : 0.0f;
    float ym = (gid.y > 0) ? phi_in[idx - stride_y] : 0.0f;
    float yp = (gid.y < p.grid_y - 1) ? phi_in[idx + stride_y] : 0.0f;
    float zm = (gid.z > 0) ? phi_in[idx - stride_z] : 0.0f;
    float zp = (gid.z < p.grid_z - 1) ? phi_in[idx + stride_z] : 0.0f;
    
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
    float restitution;           // e: coefficient of restitution (0-1)
};

kernel void particle_interactions(
    device float* particle_pos            [[buffer(0)]],  // N * 3 (read-only for positions)
    device float* particle_vel            [[buffer(1)]],  // N * 3 (read-write for velocity)
    device float* particle_excitation     [[buffer(2)]],  // N (read-write for excitation)
    device const float* particle_mass     [[buffer(3)]],  // N (read-only)
    device float* particle_heat           [[buffer(4)]],  // N (read-write for heat)
    constant ParticleInteractionParams& p [[buffer(5)]],
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
            particle_vel[j * 3 + 0],
            particle_vel[j * 3 + 1],
            particle_vel[j * 3 + 2]
        );
        float mass_j = particle_mass[j];
        float heat_j = particle_heat[j];
        
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
            float T_i = heat_i / max(mass_i, 1e-6f);
            float T_j = heat_j / max(mass_j, 1e-6f);
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
    
    // Physical constraints (non-negative)
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
    float3 local = (pos - domain_min) * inv_cell_size;
    uint3 cell = uint3(clamp(local, float3(0.0f), float3(grid_dims) - 1.0f));
    return cell;
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
    // Parameters
    constant SpatialCollisionParams& p     [[buffer(8)]],
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
                int3 neighbor = int3(cell_i) + int3(dx, dy, dz);
                
                // Boundary check
                if (neighbor.x < 0 || neighbor.x >= (int)p.grid_x ||
                    neighbor.y < 0 || neighbor.y >= (int)p.grid_y ||
                    neighbor.z < 0 || neighbor.z >= (int)p.grid_z) {
                    continue;
                }
                
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
                        particle_vel[j * 3 + 0],
                        particle_vel[j * 3 + 1],
                        particle_vel[j * 3 + 2]
                    );
                    float mass_j = particle_mass[j];
                    float heat_j = particle_heat[j];
                    
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
                    float T_i = heat_i / max(mass_i, 1e-6f);
                    float T_j = heat_j / max(mass_j, 1e-6f);
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
// Kernel: Tiled scatter with threadgroup reduction (CAS-based atomics)
// -----------------------------------------------------------------------------
// Reduces atomic contention by having each threadgroup accumulate contributions
// locally before performing a single atomic add per grid cell per threadgroup.
// This is especially effective when particles cluster spatially.
//
// Uses CAS loop for threadgroup float atomics (portable across all Metal GPUs).

kernel void scatter_particle_tiled(
    device const float* particle_pos       [[buffer(0)]],  // N * 3
    device const float* particle_mass      [[buffer(1)]],  // N
    device const float* particle_heat      [[buffer(2)]],  // N
    device atomic_float* gravity_field     [[buffer(3)]],  // X * Y * Z
    device atomic_float* heat_field        [[buffer(4)]],  // X * Y * Z
    constant TiledScatterParams& p         [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup atomic_uint* local_gravity [[threadgroup(0)]],  // grid_size uints (as float bits)
    threadgroup atomic_uint* local_heat [[threadgroup(1)]]      // grid_size uints (as float bits)
) {
    uint num_cells = p.grid_x * p.grid_y * p.grid_z;
    
    // Initialize local accumulators to 0.0f (stored as uint bits)
    uint zero_bits = as_type<uint>(0.0f);
    for (uint i = tid; i < num_cells; i += tg_size) {
        atomic_store_explicit(&local_gravity[i], zero_bits, memory_order_relaxed);
        atomic_store_explicit(&local_heat[i], zero_bits, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Scatter to local accumulators using CAS-based atomic add
    if (gid < p.num_particles) {
        float3 pos = float3(
            particle_pos[gid * 3 + 0],
            particle_pos[gid * 3 + 1],
            particle_pos[gid * 3 + 2]
        );
        float mass = particle_mass[gid];
        float heat = particle_heat[gid];
        
        // Get trilinear coordinates
        uint3 grid_dims = uint3(p.grid_x, p.grid_y, p.grid_z);
        float3 grid_pos = pos * p.inv_grid_spacing;
        grid_pos = clamp(grid_pos, float3(0.0f), float3(grid_dims) - 1.001f);
        uint3 base_idx = uint3(floor(grid_pos));
        float3 frac = grid_pos - float3(base_idx);
        
        // Compute 8 trilinear weights
        float wx0 = 1.0f - frac.x, wx1 = frac.x;
        float wy0 = 1.0f - frac.y, wy1 = frac.y;
        float wz0 = 1.0f - frac.z, wz1 = frac.z;
        
        float weights[8] = {
            wx0 * wy0 * wz0, wx0 * wy0 * wz1,
            wx0 * wy1 * wz0, wx0 * wy1 * wz1,
            wx1 * wy0 * wz0, wx1 * wy0 * wz1,
            wx1 * wy1 * wz0, wx1 * wy1 * wz1
        };
        
        uint stride_z = 1;
        uint stride_y = p.grid_z;
        uint stride_x = p.grid_y * p.grid_z;
        uint base = base_idx.x * stride_x + base_idx.y * stride_y + base_idx.z * stride_z;
        
        uint offsets[8] = {
            0, stride_z, stride_y, stride_y + stride_z,
            stride_x, stride_x + stride_z, stride_x + stride_y,
            stride_x + stride_y + stride_z
        };
        
        // Accumulate to threadgroup memory using CAS-based atomic add
        for (int i = 0; i < 8; i++) {
            uint idx = base + offsets[i];
            atomic_add_float_threadgroup(&local_gravity[idx], mass * weights[i]);
            atomic_add_float_threadgroup(&local_heat[idx], heat * weights[i]);
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Flush local accumulators to global memory (one atomic per cell per threadgroup)
    for (uint i = tid; i < num_cells; i += tg_size) {
        float gval = as_type<float>(atomic_load_explicit(&local_gravity[i], memory_order_relaxed));
        float hval = as_type<float>(atomic_load_explicit(&local_heat[i], memory_order_relaxed));
        
        if (gval != 0.0f) {
            atomic_fetch_add_explicit(&gravity_field[i], gval, memory_order_relaxed);
        }
        if (hval != 0.0f) {
            atomic_fetch_add_explicit(&heat_field[i], hval, memory_order_relaxed);
        }
    }
}

// =============================================================================
// Spectral Carrier Coupling (Resonance Potential, Langevin Flow)
// =============================================================================
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
    float temperature;             // Langevin temperature (noise strength)
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
    float sigma = max(gate_width, 1e-4f);
    return exp(-(d * d) / (sigma * sigma));
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
    uint gid [[thread_position_in_grid]]
) {
    // One thread per *existing* carrier.
    if (gid >= p.num_carriers) return;

    float cr = carrier_real[gid];
    float ci = carrier_imag[gid];
    float omega_k = carrier_omega[gid];
    float gate_w = clamp(carrier_gate_width[gid], p.gate_width_min, p.gate_width_max);
    uint state = carrier_state[gid];
    uint age = carrier_age[gid];

    // -------------------------------------------------------------------------
    // Adaptive renormalization (global, thermodynamic)
    // -------------------------------------------------------------------------
    // Use system energy scale to set a decay that self-tunes:
    //   e_scale = mean(|E|) + eps
    //   decay  = exp(-dt / e_scale)
    float mean_abs_e = energy_stats[0];
    float e_scale = max(mean_abs_e, 1e-8f);
    float adaptive_decay = exp(-p.dt / e_scale);
    float decay_mul = 1.0f;
    if (state == CARRIER_STATE_VOLATILE) decay_mul = max(p.volatile_decay_mul, 0.0f);
    else if (state == CARRIER_STATE_STABLE) decay_mul = max(p.stable_decay_mul, 0.0f);
    else decay_mul = max(p.crystallized_decay_mul, 0.0f);
    cr *= (adaptive_decay * decay_mul);
    ci *= (adaptive_decay * decay_mul);

    // Accumulate complex force F_k = Σ_i w_ik z_i, and stats for coherence + recentering.
    float force_r_raw = 0.0f;
    float force_i_raw = 0.0f;
    float w_sum = 0.0f;         // Σ w
    float w_omega_sum = 0.0f;   // Σ w ω
    float w_amp_sum = 0.0f;     // Σ w A

    // Track offender (largest weighted deviation from mean)
    uint offender_idx = 0;
    float offender_score = 0.0f;

    for (uint i = 0; i < p.num_osc; i++) {
        float omega_i = osc_omega[i];
        float amp_i = osc_amp[i];
        float phi_i = osc_phase[i];

        float t = tuning_from_freq(omega_i, omega_k, gate_w);
        // Weight includes oscillator amplitude (strong oscillators couple more).
        float w = t * amp_i;
        if (w <= p.offender_weight_floor) continue;

        // z_i = A_i e^{iθ_i}
        float zr = amp_i * cos(phi_i);
        float zi = amp_i * sin(phi_i);

        force_r_raw += w * zr;
        force_i_raw += w * zi;

        w_sum += w;
        w_omega_sum += w * omega_i;
        w_amp_sum += w * amp_i;
    }

    float mean_omega = (w_sum > 1e-8f) ? (w_omega_sum / w_sum) : omega_k;

    // Coherence / conflict:
    // - R = |Σ w z| measures phase coherence among coupled oscillators.
    // - Normalize by Σ w A to get [0,1] scale-ish.
    float R = sqrt(force_r_raw * force_r_raw + force_i_raw * force_i_raw);
    float denom = max(w_amp_sum, 1e-8f);
    float coherence = clamp(R / denom, 0.0f, 1.0f);
    float inst_conflict = 1.0f - coherence;

    // Persistent conflict (EMA) to avoid thrashing/spawn storms.
    float prev_conflict = carrier_conflict[gid];
    float a = clamp(p.ema_alpha, 0.0f, 1.0f);
    float conflict = prev_conflict * (1.0f - a) + inst_conflict * a;
    carrier_conflict[gid] = conflict;

    // Re-center ω_k toward what it's actually binding (unless crystallized).
    if (state != CARRIER_STATE_CRYSTALLIZED) {
        float rc = clamp(p.recenter_alpha, 0.0f, 1.0f);
        omega_k = omega_k * (1.0f - rc) + mean_omega * rc;
    }

    // Offender selection (phase stress): maximize weighted (1 - cos Δθ) relative to the
    // target direction (arg of the force vector).
    float psi_target = atan2(force_i_raw, force_r_raw);
    if (w_sum > 1e-8f && conflict > p.conflict_threshold) {
        for (uint i = 0; i < p.num_osc; i++) {
            float omega_i = osc_omega[i];
            float amp_i = osc_amp[i];
            float phi_i = osc_phase[i];
            float t = tuning_from_freq(omega_i, omega_k, gate_w);
            float w = t * amp_i;
            if (w <= p.offender_weight_floor) continue;
            float d = phi_i - psi_target;
            // wrap to [-pi, pi]
            d = d - 2.0f * M_PI_F * floor((d + M_PI_F) / (2.0f * M_PI_F));
            float stress = 1.0f - cos(d); // 0..2
            float score = w * stress;
            if (score > offender_score) {
                offender_score = score;
                offender_idx = i;
            }
        }
    }

    // Mean-field normalization:
    // Use the *average* tuned phasor so carrier scale doesn't explode with N.
    float inv_w = 1.0f / max(w_sum, 1e-8f);
    float force_r = force_r_raw * inv_w;
    float force_i = force_i_raw * inv_w;

    // -------------------------------------------------------------------------
    // Metabolic homeostasis for carriers (income vs expense)
    // -------------------------------------------------------------------------
    // Income: how much coherent drive the carrier receives this tick.
    // Expense: global energy scale (cost-of-living proxy).
    // If income < expense, shrink carrier amplitude; this prunes unused modes.
    float income = sqrt(force_r * force_r + force_i * force_i);
    float expense = e_scale;
    if (state != CARRIER_STATE_CRYSTALLIZED && income < expense) {
        float deficit = expense - income;
        float shrink = exp(-p.dt * deficit / (expense + 1e-8f));
        cr *= shrink;
        ci *= shrink;
    }

    // Langevin carrier update:
    //   dC = (F - λC) dt + sqrt(2T dt) * η
    float2 n = randn2(p.rng_seed ^ 0xA5A5A5A5u, gid);
    // Scale temperature gently with system energy (hotter system -> more stochasticity).
    float temp_factor = e_scale / (e_scale + 1.0f);
    float noise_scale = sqrt(max(2.0f * (p.temperature * temp_factor) * p.dt, 0.0f));
    float reg = (state == CARRIER_STATE_CRYSTALLIZED) ? 0.0f : p.carrier_reg;
    float dcr = (force_r - reg * cr) * p.dt + noise_scale * n.x;
    float dci = (force_i - reg * ci) * p.dt + noise_scale * n.y;
    cr += dcr;
    ci += dci;

    // Optional carrier-frequency repulsion (idle disambiguation mode).
    if (p.mode == 2u && p.repulsion_scale > 0.0f && p.num_carriers > 1u && state != CARRIER_STATE_CRYSTALLIZED) {
        float repel = 0.0f;
        for (uint k2 = 0; k2 < p.num_carriers; k2++) {
            if (k2 == gid) continue;
            float d = omega_k - carrier_omega[k2];
            float s = gate_w + clamp(carrier_gate_width[k2], p.gate_width_min, p.gate_width_max);
            s = max(s, 1e-3f);
            repel += d * exp(-(d * d) / (s * s));
        }
        omega_k += p.dt * p.repulsion_scale * repel;
    }

    // Crystallization state machine (volatile -> stable -> crystallized).
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

    // Anchor refresh (ε-greedy; disabled once crystallized).
    if (state != CARRIER_STATE_CRYSTALLIZED && p.num_osc > 0u) {
        uint h = hash_u32(p.rng_seed ^ (gid * 0xB4B82E39u) ^ 0x1C3A5F7Du);
        float u = u01_from_u32(h);
        uint slot = hash_u32(h + 0x3C6EF372u) % CARRIER_ANCHORS;
        uint base = gid * CARRIER_ANCHORS + slot;
        float eps_anchor = clamp(p.anchor_random_eps, 0.0f, 1.0f);
        // In exploration mode, allow weaker bonds to contribute by lowering the floor implicitly.
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

    // Conflict-driven split: spawn a new carrier centered on offender omega.
    // Never split crystallized carriers (treat as long-term memory).
    if (state != CARRIER_STATE_CRYSTALLIZED && offender_score > 0.0f) {
        // Atomically claim a new carrier slot.
        uint slot = atomic_fetch_add_explicit(out_num_carriers, 1, memory_order_relaxed);
        if (slot < p.max_carriers) {
            float omega_new = osc_omega[offender_idx];
            float amp_new = osc_amp[offender_idx];
            float phi_new = osc_phase[offender_idx];
            // Initialize new carrier from offender phasor (small magnitude to avoid shocks).
            float init_scale = 0.5f;
            float nr = init_scale * amp_new * cos(phi_new);
            float ni = init_scale * amp_new * sin(phi_new);

            carrier_real[slot] = nr;
            carrier_imag[slot] = ni;
            carrier_omega[slot] = omega_new;

            // Start moderately wide; specialization can be implemented by shrinking over time.
            carrier_gate_width[slot] = gate_w;
            carrier_conflict[slot] = 0.0f;
            carrier_state[slot] = CARRIER_STATE_VOLATILE;
            carrier_age[slot] = 0u;

            // Initialize anchors: offender as first anchor, rest empty.
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

            // Optional: record which oscillator spawned this carrier
            if (out_spawned_from_osc != nullptr) {
                out_spawned_from_osc[slot] = offender_idx;
            }

            // Randomize phase slightly by rotating c (keeps magnitude, changes ψ)
            float r = random_phases[slot] * 2.0f * M_PI_F;
            float rot_r = cos(r);
            float rot_i = sin(r);
            float rr = carrier_real[slot];
            float ri = carrier_imag[slot];
            carrier_real[slot] = rr * rot_r - ri * rot_i;
            carrier_imag[slot] = rr * rot_i + ri * rot_r;
        }

        // Reset conflict on the parent carrier after a split so it doesn't
        // repeatedly spawn every step.
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
    constant uint& num_carriers           [[buffer(12)]],
    constant SpectralCarrierParams& p     [[buffer(13)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.num_osc) return;

    float phi = osc_phase[gid];
    float omega_i = osc_omega[gid];
    float amp_i = osc_amp[gid];

    // Torque from resonance potential:
    //   θ̇_i += Σ_k T_ik (A_i R_k) sin(ψ_k - θ_i)
    float torque = 0.0f;
    for (uint k = 0; k < num_carriers; k++) {
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

    float mean_abs_e = energy_stats[0];
    float e_scale = max(mean_abs_e, 1e-8f);
    float temp_factor = e_scale / (e_scale + 1.0f);
    float noise_scale = sqrt(max(2.0f * (p.temperature * temp_factor) * p.dt, 0.0f));
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
    constant uint& num_carriers            [[buffer(5)]],
    constant SpectralCarrierParams& p      [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
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
    if (wsum <= 1e-8f) return;
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
    constant uint& num_carriers             [[buffer(16)]], // current count (read)
    constant uint& max_carriers             [[buffer(17)]],
    constant float& coupling_threshold      [[buffer(18)]], // min total coupling to be "coupled"
    constant float& gate_width_init         [[buffer(19)]],
    constant float& gate_width_min          [[buffer(20)]],
    constant float& gate_width_max          [[buffer(21)]],
    uint gid [[thread_position_in_grid]],
    constant uint& num_osc                  [[buffer(22)]]
) {
    if (gid >= num_osc) return;
    
    float omega_i = osc_omega[gid];
    float amp_i = osc_amp[gid];
    float phi_i = osc_phase[gid];
    
    // Compute total coupling weight to all existing carriers
    float total_coupling = 0.0f;
    for (uint k = 0; k < num_carriers; k++) {
        float omega_k = carrier_omega[k];
        float gate_w = clamp(carrier_gate_width[k], gate_width_min, gate_width_max);
        float t = tuning_from_freq(omega_i, omega_k, gate_w);
        total_coupling += t;
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
