#include <torch/extension.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#include <dlfcn.h>
#include <filesystem>
#include <mutex>
#include <string>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

namespace fs = std::filesystem;

namespace {

// Must match `ManifoldPhysicsParams` in `manifold_physics.metal`.
struct ManifoldPhysicsParams {
  // Grid parameters
  uint32_t num_particles;
  uint32_t grid_x;
  uint32_t grid_y;
  uint32_t grid_z;
  float grid_spacing;
  float inv_grid_spacing;
  float dt;
  
  // Fundamental physical constants
  float G;                     // Gravitational constant
  float k_B;                   // Boltzmann constant
  float sigma_SB;              // Stefan-Boltzmann constant
  
  // Material properties
  float particle_radius;
  float thermal_conductivity;
  float specific_heat;
  float dynamic_viscosity;
  float emissivity;
  float young_modulus;
};

// Must match `ManifoldFieldParams` in `manifold_physics.metal`.
struct ManifoldFieldParams {
  uint32_t grid_x;
  uint32_t grid_y;
  uint32_t grid_z;
  float grid_spacing;
  float inv_grid_spacing;
};

// Must match `ParticleGenParams` in `manifold_physics.metal`.
struct ParticleGenParams {
  uint32_t num_particles;
  float grid_x;
  float grid_y;
  float grid_z;
  float energy_scale;
  uint32_t pattern;
  float center_x;
  float center_y;
  float center_z;
  float spread;
  float dir_x;
  float dir_y;
  float dir_z;
};

// Must match `ParticleInteractionParams` in `manifold_physics.metal`.
struct ParticleInteractionParams {
  uint32_t num_particles;
  float dt;
  float particle_radius;        // r: particle radius for collision detection
  float young_modulus;          // E: Young's modulus for contact stiffness
  float thermal_conductivity;   // k: heat transfer on contact
  float restitution;            // e: coefficient of restitution (0-1)
};

// Must match `SpectralCarrierParams` in `manifold_physics.metal`.
struct SpectralCarrierParams {
  uint32_t num_osc;
  uint32_t max_carriers;
  uint32_t num_carriers;
  float dt;
  float coupling_scale;
  float carrier_reg;
  float temperature;
  uint32_t rng_seed;
  float conflict_threshold;
  float offender_weight_floor;
  float gate_width_min;
  float gate_width_max;
  float ema_alpha;
  float recenter_alpha;
  uint32_t mode;
  float anchor_random_eps;
  float stable_amp_threshold;
  float crystallize_amp_threshold;
  float crystallize_conflict_threshold;
  uint32_t crystallize_age;
  float crystallized_coupling_boost;
  float volatile_decay_mul;
  float stable_decay_mul;
  float crystallized_decay_mul;
  float topdown_phase_scale;
  float topdown_energy_scale;
  float topdown_random_energy_eps;
  float repulsion_scale;
};

// Must match `SpatialHashParams` in `manifold_physics.metal`.
struct SpatialHashParams {
  uint32_t num_particles;
  uint32_t grid_x;
  uint32_t grid_y;
  uint32_t grid_z;
  float cell_size;
  float inv_cell_size;
  float domain_min_x;
  float domain_min_y;
  float domain_min_z;
};

// Must match `SpatialCollisionParams` in `manifold_physics.metal`.
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

// Must match `TiledScatterParams` in `manifold_physics.metal`.
struct TiledScatterParams {
  uint32_t num_particles;
  uint32_t grid_x;
  uint32_t grid_y;
  uint32_t grid_z;
  float grid_spacing;
  float inv_grid_spacing;
  uint32_t tile_size;
};

// Must match `ManifoldPhysicsTextureParams` in `manifold_physics.metal`.
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

constexpr NSUInteger kThreadsPerThreadgroup = 256;

static id<MTLLibrary> g_lib = nil;
// Manifold physics pipelines
static id<MTLComputePipelineState> g_pipeline_manifold_scatter = nil;
static id<MTLComputePipelineState> g_pipeline_manifold_gather_update = nil;
static id<MTLComputePipelineState> g_pipeline_manifold_diffuse_heat = nil;
static id<MTLComputePipelineState> g_pipeline_manifold_poisson = nil;
static id<MTLComputePipelineState> g_pipeline_manifold_clear_field = nil;
static id<MTLComputePipelineState> g_pipeline_particle_interactions = nil;
// Adaptive thermodynamics (global reduction) pipelines
static id<MTLComputePipelineState> g_pipeline_reduce_float_stats_pass1 = nil;
static id<MTLComputePipelineState> g_pipeline_reduce_float_stats_finalize = nil;
// Spectral carrier (resonance potential) pipelines
static id<MTLComputePipelineState> g_pipeline_spectral_carrier_update_split = nil;
static id<MTLComputePipelineState> g_pipeline_spectral_update_osc_phases = nil;
static id<MTLComputePipelineState> g_pipeline_spectral_topdown_bias_energies = nil;
static id<MTLComputePipelineState> g_pipeline_spectral_spawn_uncoupled = nil;
// Spatial hash grid pipelines (O(N) collisions)
static id<MTLComputePipelineState> g_pipeline_spatial_hash_assign = nil;
static id<MTLComputePipelineState> g_pipeline_spatial_hash_prefix_sum = nil;
static id<MTLComputePipelineState> g_pipeline_spatial_hash_prefix_sum_parallel = nil;
static id<MTLComputePipelineState> g_pipeline_spatial_hash_scatter = nil;
static id<MTLComputePipelineState> g_pipeline_spatial_hash_collisions = nil;
// Tiled scatter pipeline (reduced atomic contention via CAS-based threadgroup atomics)
static id<MTLComputePipelineState> g_pipeline_scatter_particle_tiled = nil;
// Texture-based gather pipeline (hardware trilinear)
static id<MTLComputePipelineState> g_pipeline_gather_update_textured = nil;
// Particle generation pipelines
static id<MTLComputePipelineState> g_pipeline_generate_particle_positions = nil;
static id<MTLComputePipelineState> g_pipeline_initialize_particle_properties = nil;
static std::mutex g_pipeline_mutex;

static std::string metallib_path_for_this_module() {
  Dl_info info;
  if (dladdr((void*)&metallib_path_for_this_module, &info) == 0 || info.dli_fname == nullptr) {
    return std::string();
  }
  fs::path so_path(info.dli_fname);
  fs::path lib_path = so_path.parent_path() / "caramba_ops.metallib";
  return lib_path.string();
}

static void ensure_library_locked(id<MTLDevice> device) {
  if (g_lib != nil) {
    return;
  }

  const std::string lib_path = metallib_path_for_this_module();
  TORCH_CHECK(!lib_path.empty(), "caramba_metal_ops: failed to locate extension path via dladdr()");

  NSString* ns_path = [NSString stringWithUTF8String:lib_path.c_str()];
  NSURL* url = [NSURL fileURLWithPath:ns_path];
  NSError* err = nil;
  // newLibraryWithFile:error: is deprecated; use URL variant on newer macOS.
  g_lib = [device newLibraryWithURL:url error:&err];
  if (g_lib == nil) {
    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
    TORCH_CHECK(false, "caramba_metal_ops: failed to load metallib at ", lib_path, ": ", msg);
  }
}

static id<MTLComputePipelineState> ensure_pipeline(
    id<MTLDevice> device,
    // Important: this is a STRONG out-parameter. If left un-annotated under ARC,
    // Clang treats it as __autoreleasing and rejects passing addresses of globals.
    id<MTLComputePipelineState> __strong* pipeline,
    const char* fn_name) {
  std::lock_guard<std::mutex> lock(g_pipeline_mutex);
  ensure_library_locked(device);

  if (*pipeline != nil) {
    return *pipeline;
  }

  NSString* ns_fn = [NSString stringWithUTF8String:fn_name];
  id<MTLFunction> fn = [g_lib newFunctionWithName:ns_fn];
  TORCH_CHECK(fn != nil, "caramba_metal_ops: function `", fn_name, "` not found in metallib");

  NSError* err = nil;
  *pipeline = [device newComputePipelineStateWithFunction:fn error:&err];
  if (*pipeline == nil) {
    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
    TORCH_CHECK(false, "caramba_metal_ops: failed to create compute pipeline: ", msg);
  }

  // Basic sanity check against accidental dispatch mismatch.
  TORCH_CHECK(
      (*pipeline).maxTotalThreadsPerThreadgroup >= kThreadsPerThreadgroup,
      "caramba_metal_ops: pipeline maxTotalThreadsPerThreadgroup (",
      (int)(*pipeline).maxTotalThreadsPerThreadgroup,
      ") < expected threads (",
      (int)kThreadsPerThreadgroup,
      ")");

  return *pipeline;
}

static inline id<MTLBuffer> storage_as_mtlbuffer(const at::Tensor& t) {
  // MPS tensors are backed by MTLBuffer allocations; the storage base pointer is
  // an opaque handle that is compatible with `id<MTLBuffer>` in practice.
  //
  // NOTE: This is intentionally low-level to avoid CPU staging/copies.
  const auto& dp = t.storage().data_ptr();
  void* ctx = dp.get_context();
  TORCH_CHECK(
      ctx != nullptr,
      "caramba_metal_ops: expected MPS storage to provide an MTLBuffer context (got null). "
      "This usually indicates a non-standard tensor storage backend.");
  // Under ARC we must use a bridged cast from void* to ObjC object.
  return (__bridge id<MTLBuffer>)ctx;
}

static inline NSUInteger storage_offset_bytes(const at::Tensor& t) {
  return (NSUInteger)(t.storage_offset() * (int64_t)t.element_size());
}

static void check_contig_3d(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_contiguous(), name, ": must be contiguous");
  TORCH_CHECK(t.dim() == 3, name, ": must be 3D");
  TORCH_CHECK(t.stride(2) == 1, name, ": last dim must be contiguous (stride==1)");
}

static void check_contig_2d(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_contiguous(), name, ": must be contiguous");
  TORCH_CHECK(t.dim() == 2, name, ": must be 2D");
  TORCH_CHECK(t.stride(1) == 1, name, ": last dim must be contiguous (stride==1)");
}

static void check_contig_1d(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_contiguous(), name, ": must be contiguous");
  TORCH_CHECK(t.dim() == 1, name, ": must be 1D");
  TORCH_CHECK(t.stride(0) == 1, name, ": must be contiguous (stride==1)");
}

// ============================================================================
// Manifold Physics Kernels
// ============================================================================

void manifold_scatter_particles(
    at::Tensor particle_pos,   // (N, 3) fp32 MPS
    at::Tensor particle_mass,  // (N,) fp32 MPS
    at::Tensor particle_heat,  // (N,) fp32 MPS
    at::Tensor gravity_field,  // (X, Y, Z) fp32 MPS
    at::Tensor heat_field,     // (X, Y, Z) fp32 MPS
    double grid_spacing) {
  TORCH_CHECK(particle_pos.device().is_mps(), "manifold_scatter: particle_pos must be on MPS");
  TORCH_CHECK(particle_pos.dtype() == at::kFloat, "manifold_scatter: particle_pos must be fp32");
  TORCH_CHECK(particle_pos.is_contiguous(), "manifold_scatter: particle_pos must be contiguous");
  TORCH_CHECK(particle_pos.dim() == 2 && particle_pos.size(1) == 3, "manifold_scatter: particle_pos must be (N, 3)");

  TORCH_CHECK(particle_mass.is_contiguous() && particle_mass.dim() == 1, "manifold_scatter: particle_mass must be contiguous 1D");
  TORCH_CHECK(particle_heat.is_contiguous() && particle_heat.dim() == 1, "manifold_scatter: particle_heat must be contiguous 1D");
  TORCH_CHECK(gravity_field.is_contiguous() && gravity_field.dim() == 3, "manifold_scatter: gravity_field must be contiguous 3D");
  TORCH_CHECK(heat_field.is_contiguous() && heat_field.dim() == 3, "manifold_scatter: heat_field must be contiguous 3D");

  const int64_t N = particle_pos.size(0);
  const int64_t gx = gravity_field.size(0);
  const int64_t gy = gravity_field.size(1);
  const int64_t gz = gravity_field.size(2);

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "manifold_scatter: failed to get current MPS stream");
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "manifold_scatter: failed to get MTLComputeCommandEncoder");

  id<MTLComputePipelineState> pipeline = ensure_pipeline(device, &g_pipeline_manifold_scatter, "scatter_particle");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(particle_pos, 0);
  set_tensor(particle_mass, 1);
  set_tensor(particle_heat, 2);
  set_tensor(gravity_field, 3);
  set_tensor(heat_field, 4);

  ManifoldPhysicsParams prm;
  prm.num_particles = (uint32_t)N;
  prm.grid_x = (uint32_t)gx;
  prm.grid_y = (uint32_t)gy;
  prm.grid_z = (uint32_t)gz;
  prm.grid_spacing = (float)grid_spacing;
  prm.inv_grid_spacing = 1.0f / (float)grid_spacing;
  [encoder setBytes:&prm length:sizeof(ManifoldPhysicsParams) atIndex:5];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const NSUInteger num_groups = (N + kThreadsPerThreadgroup - 1) / kThreadsPerThreadgroup;
  const MTLSize grid = MTLSizeMake(num_groups, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  stream->endKernelCoalescing();
}

void manifold_gather_update_particles(
    at::Tensor gravity_potential,  // (X, Y, Z) fp32 MPS
    at::Tensor temperature_field,  // (X, Y, Z) fp32 MPS
    at::Tensor particle_pos,       // (N, 3) fp32 MPS in/out
    at::Tensor particle_vel,       // (N, 3) fp32 MPS in/out
    at::Tensor particle_energy,    // (N,) fp32 MPS in/out
    at::Tensor particle_heat,      // (N,) fp32 MPS in/out
    at::Tensor particle_excitation,// (N,) fp32 MPS in/out
    at::Tensor particle_mass,      // (N,) fp32 MPS
    double dt,
    double grid_spacing,
    // Fundamental constants
    double G,
    double k_B,
    double sigma_SB,
    // Material properties
    double particle_radius,
    double thermal_conductivity,
    double specific_heat,
    double dynamic_viscosity,
    double emissivity,
    double young_modulus) {
  TORCH_CHECK(gravity_potential.device().is_mps(), "manifold_gather_update: gravity_potential must be on MPS");
  TORCH_CHECK(gravity_potential.is_contiguous() && gravity_potential.dim() == 3, "manifold_gather_update: gravity_potential must be contiguous 3D");
  TORCH_CHECK(particle_pos.is_contiguous() && particle_pos.dim() == 2 && particle_pos.size(1) == 3, "manifold_gather_update: particle_pos must be (N, 3)");

  const int64_t N = particle_pos.size(0);
  const int64_t gx = gravity_potential.size(0);
  const int64_t gy = gravity_potential.size(1);
  const int64_t gz = gravity_potential.size(2);

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "manifold_gather_update: failed to get current MPS stream");
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "manifold_gather_update: failed to get MTLComputeCommandEncoder");

  id<MTLComputePipelineState> pipeline = ensure_pipeline(device, &g_pipeline_manifold_gather_update, "gather_update_particles");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(gravity_potential, 0);
  set_tensor(temperature_field, 1);
  set_tensor(particle_pos, 2);
  set_tensor(particle_vel, 3);
  set_tensor(particle_energy, 4);
  set_tensor(particle_heat, 5);
  set_tensor(particle_excitation, 6);
  set_tensor(particle_mass, 7);

  ManifoldPhysicsParams prm;
  prm.num_particles = (uint32_t)N;
  prm.grid_x = (uint32_t)gx;
  prm.grid_y = (uint32_t)gy;
  prm.grid_z = (uint32_t)gz;
  prm.grid_spacing = (float)grid_spacing;
  prm.inv_grid_spacing = 1.0f / (float)grid_spacing;
  prm.dt = (float)dt;
  prm.G = (float)G;
  prm.k_B = (float)k_B;
  prm.sigma_SB = (float)sigma_SB;
  prm.particle_radius = (float)particle_radius;
  prm.thermal_conductivity = (float)thermal_conductivity;
  prm.specific_heat = (float)specific_heat;
  prm.dynamic_viscosity = (float)dynamic_viscosity;
  prm.emissivity = (float)emissivity;
  prm.young_modulus = (float)young_modulus;
  [encoder setBytes:&prm length:sizeof(ManifoldPhysicsParams) atIndex:8];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const NSUInteger num_groups = (N + kThreadsPerThreadgroup - 1) / kThreadsPerThreadgroup;
  const MTLSize grid = MTLSizeMake(num_groups, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  stream->endKernelCoalescing();
}

void manifold_diffuse_heat(
    at::Tensor temp_in,   // (X, Y, Z) fp32 MPS
    at::Tensor temp_out,  // (X, Y, Z) fp32 MPS
    double diffusion_coef,
    double dt,
    double grid_spacing) {
  TORCH_CHECK(temp_in.device().is_mps(), "manifold_diffuse_heat: temp_in must be on MPS");
  TORCH_CHECK(temp_in.is_contiguous() && temp_in.dim() == 3, "manifold_diffuse_heat: temp_in must be contiguous 3D");
  TORCH_CHECK(temp_out.is_contiguous() && temp_out.dim() == 3, "manifold_diffuse_heat: temp_out must be contiguous 3D");

  const int64_t gx = temp_in.size(0);
  const int64_t gy = temp_in.size(1);
  const int64_t gz = temp_in.size(2);

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();

  id<MTLComputePipelineState> pipeline = ensure_pipeline(device, &g_pipeline_manifold_diffuse_heat, "diffuse_heat_field");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(temp_in, 0);
  set_tensor(temp_out, 1);

  ManifoldFieldParams prm;
  prm.grid_x = (uint32_t)gx;
  prm.grid_y = (uint32_t)gy;
  prm.grid_z = (uint32_t)gz;
  prm.grid_spacing = (float)grid_spacing;
  prm.inv_grid_spacing = 1.0f / (float)grid_spacing;
  [encoder setBytes:&prm length:sizeof(ManifoldFieldParams) atIndex:2];

  float diff_coef_f = (float)diffusion_coef;
  float dt_f = (float)dt;
  [encoder setBytes:&diff_coef_f length:sizeof(float) atIndex:3];
  [encoder setBytes:&dt_f length:sizeof(float) atIndex:4];

  // Dispatch 3D grid
  const MTLSize tg = MTLSizeMake(8, 8, 4);  // 256 threads total
  const MTLSize grid = MTLSizeMake(
      (gx + 7) / 8,
      (gy + 7) / 8,
      (gz + 3) / 4
  );
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  stream->endKernelCoalescing();
}

void manifold_poisson_step(
    at::Tensor phi_in,   // (X, Y, Z) fp32 MPS
    at::Tensor rho,      // (X, Y, Z) fp32 MPS
    at::Tensor phi_out,  // (X, Y, Z) fp32 MPS
    double gravity_4pi,
    double grid_spacing) {
  TORCH_CHECK(phi_in.device().is_mps(), "manifold_poisson_step: phi_in must be on MPS");
  TORCH_CHECK(phi_in.is_contiguous() && phi_in.dim() == 3, "manifold_poisson_step: phi_in must be contiguous 3D");
  TORCH_CHECK(rho.is_contiguous() && rho.dim() == 3, "manifold_poisson_step: rho must be contiguous 3D");
  TORCH_CHECK(phi_out.is_contiguous() && phi_out.dim() == 3, "manifold_poisson_step: phi_out must be contiguous 3D");

  const int64_t gx = phi_in.size(0);
  const int64_t gy = phi_in.size(1);
  const int64_t gz = phi_in.size(2);

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();

  id<MTLComputePipelineState> pipeline = ensure_pipeline(device, &g_pipeline_manifold_poisson, "poisson_jacobi_step");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(phi_in, 0);
  set_tensor(rho, 1);
  set_tensor(phi_out, 2);

  ManifoldFieldParams prm;
  prm.grid_x = (uint32_t)gx;
  prm.grid_y = (uint32_t)gy;
  prm.grid_z = (uint32_t)gz;
  prm.grid_spacing = (float)grid_spacing;
  prm.inv_grid_spacing = 1.0f / (float)grid_spacing;
  [encoder setBytes:&prm length:sizeof(ManifoldFieldParams) atIndex:3];

  float gravity_4pi_f = (float)gravity_4pi;
  [encoder setBytes:&gravity_4pi_f length:sizeof(float) atIndex:4];

  const MTLSize tg = MTLSizeMake(8, 8, 4);
  const MTLSize grid = MTLSizeMake(
      (gx + 7) / 8,
      (gy + 7) / 8,
      (gz + 3) / 4
  );
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  stream->endKernelCoalescing();
}

void manifold_clear_field(at::Tensor field) {
  TORCH_CHECK(field.device().is_mps(), "manifold_clear_field: field must be on MPS");
  TORCH_CHECK(field.is_contiguous(), "manifold_clear_field: field must be contiguous");

  const int64_t num_elements = field.numel();

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();

  id<MTLComputePipelineState> pipeline = ensure_pipeline(device, &g_pipeline_manifold_clear_field, "clear_field");
  [encoder setComputePipelineState:pipeline];

  id<MTLBuffer> buf = storage_as_mtlbuffer(field);
  const NSUInteger off = storage_offset_bytes(field);
  [encoder setBuffer:buf offset:off atIndex:0];

  uint32_t n = (uint32_t)num_elements;
  [encoder setBytes:&n length:sizeof(uint32_t) atIndex:1];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const NSUInteger num_groups = (num_elements + kThreadsPerThreadgroup - 1) / kThreadsPerThreadgroup;
  const MTLSize grid = MTLSizeMake(num_groups, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  stream->endKernelCoalescing();
}

void particle_interactions(
    at::Tensor particle_pos,       // (N, 3) fp32 MPS
    at::Tensor particle_vel,       // (N, 3) fp32 MPS in/out
    at::Tensor particle_excitation,// (N,) fp32 MPS in/out
    at::Tensor particle_mass,      // (N,) fp32 MPS
    at::Tensor particle_heat,      // (N,) fp32 MPS in/out - heat from collisions
    double dt,
    double particle_radius,
    double young_modulus,
    double thermal_conductivity,
    double restitution) {
  
  const int64_t N = particle_pos.size(0);
  if (N == 0) return;
  
  TORCH_CHECK(particle_pos.device().is_mps(), "particle_interactions: particle_pos must be on MPS");
  TORCH_CHECK(particle_pos.is_contiguous() && particle_pos.dim() == 2 && particle_pos.size(1) == 3,
              "particle_interactions: particle_pos must be contiguous (N, 3)");
  TORCH_CHECK(particle_vel.is_contiguous() && particle_vel.dim() == 2 && particle_vel.size(1) == 3,
              "particle_interactions: particle_vel must be contiguous (N, 3)");
  TORCH_CHECK(particle_excitation.is_contiguous() && particle_excitation.dim() == 1,
              "particle_interactions: particle_excitation must be contiguous 1D");
  TORCH_CHECK(particle_heat.is_contiguous() && particle_heat.dim() == 1,
              "particle_interactions: particle_heat must be contiguous 1D");
  
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  
  id<MTLComputePipelineState> pipeline = ensure_pipeline(device, &g_pipeline_particle_interactions, "particle_interactions");
  [encoder setComputePipelineState:pipeline];
  
  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };
  
  set_tensor(particle_pos, 0);
  set_tensor(particle_vel, 1);
  set_tensor(particle_excitation, 2);
  set_tensor(particle_mass, 3);
  set_tensor(particle_heat, 4);
  
  ParticleInteractionParams prm;
  prm.num_particles = (uint32_t)N;
  prm.dt = (float)dt;
  prm.particle_radius = (float)particle_radius;
  prm.young_modulus = (float)young_modulus;
  prm.thermal_conductivity = (float)thermal_conductivity;
  prm.restitution = (float)restitution;
  [encoder setBytes:&prm length:sizeof(ParticleInteractionParams) atIndex:5];
  
  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const NSUInteger num_groups = (N + kThreadsPerThreadgroup - 1) / kThreadsPerThreadgroup;
  const MTLSize grid = MTLSizeMake(num_groups, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  
  stream->endKernelCoalescing();
}

// ============================================================================
// Adaptive Thermodynamics: reduction for global energy statistics
// ============================================================================
// Computes out_stats (4 floats): [mean_abs, mean, std, count]
//
// This stays entirely on the GPU. No CPU sync required.
void thermo_reduce_energy_stats(
    at::Tensor x,         // (N,) fp32 MPS
    at::Tensor out_stats  // (4,) fp32 MPS (output)
) {
  TORCH_CHECK(x.device().is_mps(), "thermo_reduce_energy_stats: x must be on MPS");
  TORCH_CHECK(out_stats.device().is_mps(), "thermo_reduce_energy_stats: out_stats must be on MPS");
  TORCH_CHECK(x.dtype() == at::kFloat, "thermo_reduce_energy_stats: x must be fp32");
  TORCH_CHECK(out_stats.dtype() == at::kFloat, "thermo_reduce_energy_stats: out_stats must be fp32");
  check_contig_1d(x, "thermo_reduce_energy_stats: x");
  check_contig_1d(out_stats, "thermo_reduce_energy_stats: out_stats");
  TORCH_CHECK(out_stats.numel() == 4, "thermo_reduce_energy_stats: out_stats must have 4 elements");

  const int64_t N = x.numel();
  if (N <= 0) {
    // Define a sane empty result: zeros + count=0
    out_stats.zero_();
    return;
  }

  const NSUInteger num_groups = (N + (int64_t)kThreadsPerThreadgroup - 1) / kThreadsPerThreadgroup;

  // group_stats: (num_groups, 4) contiguous fp32 on MPS
  at::Tensor group_stats = at::empty({(int64_t)num_groups, 4}, x.options());

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "thermo_reduce_energy_stats: failed to get MTLComputeCommandEncoder");

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  // Pass 1: per-threadgroup partial sums
  {
    id<MTLComputePipelineState> pipeline =
        ensure_pipeline(device, &g_pipeline_reduce_float_stats_pass1, "reduce_float_stats_pass1");
    [encoder setComputePipelineState:pipeline];
    set_tensor(x, 0);
    set_tensor(group_stats, 1);
    uint32_t n_u = (uint32_t)N;
    [encoder setBytes:&n_u length:sizeof(uint32_t) atIndex:2];

    const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
    const MTLSize grid = MTLSizeMake(num_groups, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  }

  // Pass 2: finalize to a single stats vector
  {
    id<MTLComputePipelineState> pipeline =
        ensure_pipeline(device, &g_pipeline_reduce_float_stats_finalize, "reduce_float_stats_finalize");
    [encoder setComputePipelineState:pipeline];
    set_tensor(group_stats, 0);
    set_tensor(out_stats, 1);
    uint32_t g_u = (uint32_t)num_groups;
    [encoder setBytes:&g_u length:sizeof(uint32_t) atIndex:2];

    const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
    const MTLSize grid = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  }

  stream->endKernelCoalescing();
}

// ============================================================================
// Tiled Scatter (reduced atomic contention)
// ============================================================================

void scatter_particle_tiled(
    at::Tensor particle_pos,       // (N, 3) fp32 MPS
    at::Tensor particle_mass,      // (N,) fp32 MPS
    at::Tensor particle_heat,      // (N,) fp32 MPS
    at::Tensor gravity_field,      // (X, Y, Z) fp32 MPS atomic
    at::Tensor heat_field,         // (X, Y, Z) fp32 MPS atomic
    int64_t grid_x,
    int64_t grid_y,
    int64_t grid_z,
    double grid_spacing) {
  
  const int64_t N = particle_pos.size(0);
  if (N == 0) return;
  
  TORCH_CHECK(particle_pos.device().is_mps(), "scatter_particle_tiled: particle_pos must be on MPS");
  
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  
  id<MTLComputePipelineState> pipeline = ensure_pipeline(device, &g_pipeline_scatter_particle_tiled, "scatter_particle_tiled");
  [encoder setComputePipelineState:pipeline];
  
  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };
  
  set_tensor(particle_pos, 0);
  set_tensor(particle_mass, 1);
  set_tensor(particle_heat, 2);
  set_tensor(gravity_field, 3);
  set_tensor(heat_field, 4);
  
  TiledScatterParams prm;
  prm.num_particles = (uint32_t)N;
  prm.grid_x = (uint32_t)grid_x;
  prm.grid_y = (uint32_t)grid_y;
  prm.grid_z = (uint32_t)grid_z;
  prm.grid_spacing = (float)grid_spacing;
  prm.inv_grid_spacing = 1.0f / (float)grid_spacing;
  prm.tile_size = kThreadsPerThreadgroup;
  [encoder setBytes:&prm length:sizeof(TiledScatterParams) atIndex:5];
  
  // Allocate threadgroup memory for local accumulators
  uint64_t num_cells = grid_x * grid_y * grid_z;
  uint64_t tg_mem_size = num_cells * sizeof(uint32_t);  // atomic_uint per cell
  [encoder setThreadgroupMemoryLength:tg_mem_size atIndex:0];  // local_gravity
  [encoder setThreadgroupMemoryLength:tg_mem_size atIndex:1];  // local_heat
  
  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const NSUInteger num_groups = (N + kThreadsPerThreadgroup - 1) / kThreadsPerThreadgroup;
  const MTLSize grid = MTLSizeMake(num_groups, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  
  stream->endKernelCoalescing();
}

// ============================================================================
// Spatial Hash Grid Acceleration (O(N) collisions)
// ============================================================================

void spatial_hash_assign(
    at::Tensor particle_pos,       // (N, 3) fp32 MPS
    at::Tensor particle_cell_idx,  // (N,) uint32 MPS out
    at::Tensor cell_counts,        // (num_cells,) uint32 MPS out (atomic)
    int64_t grid_x,
    int64_t grid_y,
    int64_t grid_z,
    double cell_size,
    double domain_min_x,
    double domain_min_y,
    double domain_min_z) {
  
  const int64_t N = particle_pos.size(0);
  if (N == 0) return;
  
  TORCH_CHECK(particle_pos.device().is_mps(), "spatial_hash_assign: particle_pos must be on MPS");
  
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  
  id<MTLComputePipelineState> pipeline = ensure_pipeline(device, &g_pipeline_spatial_hash_assign, "spatial_hash_assign");
  [encoder setComputePipelineState:pipeline];
  
  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };
  
  set_tensor(particle_pos, 0);
  set_tensor(particle_cell_idx, 1);
  set_tensor(cell_counts, 2);
  
  SpatialHashParams prm;
  prm.num_particles = (uint32_t)N;
  prm.grid_x = (uint32_t)grid_x;
  prm.grid_y = (uint32_t)grid_y;
  prm.grid_z = (uint32_t)grid_z;
  prm.cell_size = (float)cell_size;
  prm.inv_cell_size = 1.0f / (float)cell_size;
  prm.domain_min_x = (float)domain_min_x;
  prm.domain_min_y = (float)domain_min_y;
  prm.domain_min_z = (float)domain_min_z;
  [encoder setBytes:&prm length:sizeof(SpatialHashParams) atIndex:3];
  
  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const NSUInteger num_groups = (N + kThreadsPerThreadgroup - 1) / kThreadsPerThreadgroup;
  const MTLSize grid = MTLSizeMake(num_groups, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  
  stream->endKernelCoalescing();
}

void spatial_hash_prefix_sum(
    at::Tensor cell_counts,        // (num_cells,) uint32 MPS
    at::Tensor cell_starts,        // (num_cells + 1,) uint32 MPS out
    int64_t num_cells) {
  
  if (num_cells == 0) return;
  
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  
  id<MTLComputePipelineState> pipeline = ensure_pipeline(device, &g_pipeline_spatial_hash_prefix_sum, "spatial_hash_prefix_sum");
  [encoder setComputePipelineState:pipeline];
  
  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };
  
  set_tensor(cell_counts, 0);
  set_tensor(cell_starts, 1);
  
  uint32_t nc = (uint32_t)num_cells;
  [encoder setBytes:&nc length:sizeof(uint32_t) atIndex:2];
  
  // Single thread for sequential scan
  const MTLSize tg = MTLSizeMake(1, 1, 1);
  const MTLSize grid = MTLSizeMake(1, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  
  stream->endKernelCoalescing();
}

void spatial_hash_scatter(
    at::Tensor particle_cell_idx,  // (N,) uint32 MPS
    at::Tensor sorted_particle_idx,// (N,) uint32 MPS out
    at::Tensor cell_offsets,       // (num_cells,) uint32 MPS (working copy of cell_starts)
    int64_t num_particles) {
  
  if (num_particles == 0) return;
  
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  
  id<MTLComputePipelineState> pipeline = ensure_pipeline(device, &g_pipeline_spatial_hash_scatter, "spatial_hash_scatter");
  [encoder setComputePipelineState:pipeline];
  
  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };
  
  set_tensor(particle_cell_idx, 0);
  set_tensor(sorted_particle_idx, 1);
  set_tensor(cell_offsets, 2);
  
  uint32_t np = (uint32_t)num_particles;
  [encoder setBytes:&np length:sizeof(uint32_t) atIndex:3];
  
  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const NSUInteger num_groups = (num_particles + kThreadsPerThreadgroup - 1) / kThreadsPerThreadgroup;
  const MTLSize grid = MTLSizeMake(num_groups, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  
  stream->endKernelCoalescing();
}

void spatial_hash_collisions(
    at::Tensor particle_pos,       // (N, 3) fp32 MPS
    at::Tensor particle_vel,       // (N, 3) fp32 MPS in/out
    at::Tensor particle_excitation,// (N,) fp32 MPS in/out
    at::Tensor particle_mass,      // (N,) fp32 MPS
    at::Tensor particle_heat,      // (N,) fp32 MPS in/out
    at::Tensor sorted_particle_idx,// (N,) uint32 MPS
    at::Tensor cell_starts,        // (num_cells + 1,) uint32 MPS
    at::Tensor particle_cell_idx,  // (N,) uint32 MPS
    int64_t grid_x,
    int64_t grid_y,
    int64_t grid_z,
    double cell_size,
    double domain_min_x,
    double domain_min_y,
    double domain_min_z,
    double dt,
    double particle_radius,
    double young_modulus,
    double thermal_conductivity,
    double restitution) {
  
  const int64_t N = particle_pos.size(0);
  if (N == 0) return;
  
  TORCH_CHECK(particle_pos.device().is_mps(), "spatial_hash_collisions: particle_pos must be on MPS");
  
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  
  id<MTLComputePipelineState> pipeline = ensure_pipeline(device, &g_pipeline_spatial_hash_collisions, "spatial_hash_collisions");
  [encoder setComputePipelineState:pipeline];
  
  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };
  
  set_tensor(particle_pos, 0);
  set_tensor(particle_vel, 1);
  set_tensor(particle_excitation, 2);
  set_tensor(particle_mass, 3);
  set_tensor(particle_heat, 4);
  set_tensor(sorted_particle_idx, 5);
  set_tensor(cell_starts, 6);
  set_tensor(particle_cell_idx, 7);
  
  SpatialCollisionParams prm;
  prm.num_particles = (uint32_t)N;
  prm.grid_x = (uint32_t)grid_x;
  prm.grid_y = (uint32_t)grid_y;
  prm.grid_z = (uint32_t)grid_z;
  prm.cell_size = (float)cell_size;
  prm.inv_cell_size = 1.0f / (float)cell_size;
  prm.domain_min_x = (float)domain_min_x;
  prm.domain_min_y = (float)domain_min_y;
  prm.domain_min_z = (float)domain_min_z;
  prm.dt = (float)dt;
  prm.particle_radius = (float)particle_radius;
  prm.young_modulus = (float)young_modulus;
  prm.thermal_conductivity = (float)thermal_conductivity;
  prm.restitution = (float)restitution;
  [encoder setBytes:&prm length:sizeof(SpatialCollisionParams) atIndex:8];
  
  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const NSUInteger num_groups = (N + kThreadsPerThreadgroup - 1) / kThreadsPerThreadgroup;
  const MTLSize grid = MTLSizeMake(num_groups, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  
  stream->endKernelCoalescing();
}

// ============================================================================
// Spectral Carrier Coupling (Resonance Potential, Langevin Flow)
// ============================================================================

void spectral_carrier_update_and_split(
    at::Tensor osc_phase,            // (N,) fp32 MPS
    at::Tensor osc_omega,            // (N,) fp32 MPS
    at::Tensor osc_amp,              // (N,) fp32 MPS
    at::Tensor carrier_real,         // (maxM,) fp32 MPS in/out
    at::Tensor carrier_imag,         // (maxM,) fp32 MPS in/out
    at::Tensor carrier_omega,        // (maxM,) fp32 MPS in/out
    at::Tensor carrier_gate_width,   // (maxM,) fp32 MPS in/out
    at::Tensor carrier_conflict,     // (maxM,) fp32 MPS out
    at::Tensor carrier_state,        // (maxM,) int32 MPS in/out
    at::Tensor carrier_age,          // (maxM,) int32 MPS in/out
    at::Tensor carrier_anchor_idx,   // (maxM*anchors,) int32 MPS in/out
    at::Tensor carrier_anchor_phase, // (maxM*anchors,) fp32 MPS in/out
    at::Tensor carrier_anchor_weight,// (maxM*anchors,) fp32 MPS in/out
    at::Tensor num_carriers,         // (1,) int32 MPS in/out (atomic counter)
    at::Tensor spawned_from_osc,     // (maxM,) int32 MPS out
    at::Tensor random_phases,        // (maxM,) fp32 MPS (uniform [0,1])
    at::Tensor energy_stats,         // (4,) fp32 MPS (mean_abs, mean, std, count)
    int64_t current_carriers,
    int64_t max_carriers,
    double dt,
    double coupling_scale,
    double carrier_reg,
    double temperature,
    uint32_t rng_seed,
    double conflict_threshold,
    double offender_weight_floor,
    double gate_width_min,
    double gate_width_max,
    double ema_alpha,
    double recenter_alpha,
    int64_t mode,
    double anchor_random_eps,
    double stable_amp_threshold,
    double crystallize_amp_threshold,
    double crystallize_conflict_threshold,
    int64_t crystallize_age,
    double crystallized_coupling_boost,
    double volatile_decay_mul,
    double stable_decay_mul,
    double crystallized_decay_mul,
    double topdown_phase_scale,
    double topdown_energy_scale,
    double topdown_random_energy_eps,
    double repulsion_scale) {

  check_contig_1d(osc_phase, "spectral_carrier_update: osc_phase");
  check_contig_1d(osc_omega, "spectral_carrier_update: osc_omega");
  check_contig_1d(osc_amp, "spectral_carrier_update: osc_amp");
  check_contig_1d(carrier_real, "spectral_carrier_update: carrier_real");
  check_contig_1d(carrier_imag, "spectral_carrier_update: carrier_imag");
  check_contig_1d(carrier_omega, "spectral_carrier_update: carrier_omega");
  check_contig_1d(carrier_gate_width, "spectral_carrier_update: carrier_gate_width");
  check_contig_1d(carrier_conflict, "spectral_carrier_update: carrier_conflict");
  check_contig_1d(carrier_state, "spectral_carrier_update: carrier_state");
  check_contig_1d(carrier_age, "spectral_carrier_update: carrier_age");
  check_contig_1d(carrier_anchor_idx, "spectral_carrier_update: carrier_anchor_idx");
  check_contig_1d(carrier_anchor_phase, "spectral_carrier_update: carrier_anchor_phase");
  check_contig_1d(carrier_anchor_weight, "spectral_carrier_update: carrier_anchor_weight");
  check_contig_1d(num_carriers, "spectral_carrier_update: num_carriers");
  check_contig_1d(spawned_from_osc, "spectral_carrier_update: spawned_from_osc");
  check_contig_1d(random_phases, "spectral_carrier_update: random_phases");
  check_contig_1d(energy_stats, "spectral_carrier_update: energy_stats");

  TORCH_CHECK(osc_phase.device().is_mps(), "spectral_carrier_update: osc_phase must be on MPS");
  TORCH_CHECK(osc_phase.dtype() == at::kFloat, "spectral_carrier_update: osc_phase must be fp32");
  TORCH_CHECK(num_carriers.dtype() == at::kInt, "spectral_carrier_update: num_carriers must be int32");
  TORCH_CHECK(carrier_state.dtype() == at::kInt, "spectral_carrier_update: carrier_state must be int32");
  TORCH_CHECK(carrier_age.dtype() == at::kInt, "spectral_carrier_update: carrier_age must be int32");
  TORCH_CHECK(carrier_anchor_idx.dtype() == at::kInt, "spectral_carrier_update: carrier_anchor_idx must be int32");
  TORCH_CHECK(energy_stats.device().is_mps(), "spectral_carrier_update: energy_stats must be on MPS");
  TORCH_CHECK(energy_stats.dtype() == at::kFloat, "spectral_carrier_update: energy_stats must be fp32");
  TORCH_CHECK(energy_stats.numel() == 4, "spectral_carrier_update: energy_stats must have 4 elements");

  const int64_t N = osc_phase.size(0);
  if (N == 0 || current_carriers == 0) return;

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();

  id<MTLComputePipelineState> pipeline =
      ensure_pipeline(device, &g_pipeline_spectral_carrier_update_split, "spectral_carrier_update_and_split");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(osc_phase, 0);
  set_tensor(osc_omega, 1);
  set_tensor(osc_amp, 2);
  set_tensor(carrier_real, 3);
  set_tensor(carrier_imag, 4);
  set_tensor(carrier_omega, 5);
  set_tensor(carrier_gate_width, 6);
  set_tensor(carrier_conflict, 7);
  set_tensor(carrier_state, 8);
  set_tensor(carrier_age, 9);
  set_tensor(carrier_anchor_idx, 10);
  set_tensor(carrier_anchor_phase, 11);
  set_tensor(carrier_anchor_weight, 12);
  set_tensor(num_carriers, 13);
  set_tensor(spawned_from_osc, 14);
  set_tensor(random_phases, 15);
  set_tensor(energy_stats, 16);

  SpectralCarrierParams prm;
  prm.num_osc = (uint32_t)N;
  prm.max_carriers = (uint32_t)max_carriers;
  prm.num_carriers = (uint32_t)current_carriers;
  prm.dt = (float)dt;
  prm.coupling_scale = (float)coupling_scale;
  prm.carrier_reg = (float)carrier_reg;
  prm.temperature = (float)temperature;
  prm.rng_seed = (uint32_t)rng_seed;
  prm.conflict_threshold = (float)conflict_threshold;
  prm.offender_weight_floor = (float)offender_weight_floor;
  prm.gate_width_min = (float)gate_width_min;
  prm.gate_width_max = (float)gate_width_max;
  prm.ema_alpha = (float)ema_alpha;
  prm.recenter_alpha = (float)recenter_alpha;
  prm.mode = (uint32_t)mode;
  prm.anchor_random_eps = (float)anchor_random_eps;
  prm.stable_amp_threshold = (float)stable_amp_threshold;
  prm.crystallize_amp_threshold = (float)crystallize_amp_threshold;
  prm.crystallize_conflict_threshold = (float)crystallize_conflict_threshold;
  prm.crystallize_age = (uint32_t)crystallize_age;
  prm.crystallized_coupling_boost = (float)crystallized_coupling_boost;
  prm.volatile_decay_mul = (float)volatile_decay_mul;
  prm.stable_decay_mul = (float)stable_decay_mul;
  prm.crystallized_decay_mul = (float)crystallized_decay_mul;
  prm.topdown_phase_scale = (float)topdown_phase_scale;
  prm.topdown_energy_scale = (float)topdown_energy_scale;
  prm.topdown_random_energy_eps = (float)topdown_random_energy_eps;
  prm.repulsion_scale = (float)repulsion_scale;
  [encoder setBytes:&prm length:sizeof(SpectralCarrierParams) atIndex:17];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const NSUInteger num_groups = (current_carriers + kThreadsPerThreadgroup - 1) / kThreadsPerThreadgroup;
  const MTLSize grid = MTLSizeMake(num_groups, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  stream->endKernelCoalescing();
}

void spectral_update_oscillator_phases(
    at::Tensor osc_phase,            // (N,) fp32 MPS in/out
    at::Tensor osc_omega,            // (N,) fp32 MPS
    at::Tensor osc_amp,              // (N,) fp32 MPS
    at::Tensor carrier_real,         // (maxM,) fp32 MPS
    at::Tensor carrier_imag,         // (maxM,) fp32 MPS
    at::Tensor carrier_omega,        // (maxM,) fp32 MPS
    at::Tensor carrier_gate_width,   // (maxM,) fp32 MPS
    at::Tensor carrier_state,        // (maxM,) int32 MPS
    at::Tensor carrier_anchor_idx,   // (maxM*anchors,) int32 MPS
    at::Tensor carrier_anchor_phase, // (maxM*anchors,) fp32 MPS
    at::Tensor carrier_anchor_weight,// (maxM*anchors,) fp32 MPS
    at::Tensor energy_stats,         // (4,) fp32 MPS (mean_abs, mean, std, count)
    int64_t num_carriers_i,
    int64_t max_carriers,
    double dt,
    double coupling_scale,
    double temperature,
    uint32_t rng_seed,
    double gate_width_min,
    double gate_width_max,
    double crystallized_coupling_boost,
    double topdown_phase_scale) {

  check_contig_1d(osc_phase, "spectral_update_osc: osc_phase");
  check_contig_1d(osc_omega, "spectral_update_osc: osc_omega");
  check_contig_1d(osc_amp, "spectral_update_osc: osc_amp");
  check_contig_1d(carrier_real, "spectral_update_osc: carrier_real");
  check_contig_1d(carrier_imag, "spectral_update_osc: carrier_imag");
  check_contig_1d(carrier_omega, "spectral_update_osc: carrier_omega");
  check_contig_1d(carrier_gate_width, "spectral_update_osc: carrier_gate_width");
  check_contig_1d(carrier_state, "spectral_update_osc: carrier_state");
  check_contig_1d(carrier_anchor_idx, "spectral_update_osc: carrier_anchor_idx");
  check_contig_1d(carrier_anchor_phase, "spectral_update_osc: carrier_anchor_phase");
  check_contig_1d(carrier_anchor_weight, "spectral_update_osc: carrier_anchor_weight");
  check_contig_1d(energy_stats, "spectral_update_osc: energy_stats");
  TORCH_CHECK(energy_stats.device().is_mps(), "spectral_update_osc: energy_stats must be on MPS");
  TORCH_CHECK(energy_stats.dtype() == at::kFloat, "spectral_update_osc: energy_stats must be fp32");
  TORCH_CHECK(energy_stats.numel() == 4, "spectral_update_osc: energy_stats must have 4 elements");
  TORCH_CHECK(carrier_state.dtype() == at::kInt, "spectral_update_osc: carrier_state must be int32");
  TORCH_CHECK(carrier_anchor_idx.dtype() == at::kInt, "spectral_update_osc: carrier_anchor_idx must be int32");

  const int64_t N = osc_phase.size(0);
  if (N == 0 || num_carriers_i == 0) return;

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();

  id<MTLComputePipelineState> pipeline =
      ensure_pipeline(device, &g_pipeline_spectral_update_osc_phases, "spectral_update_oscillator_phases");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(osc_phase, 0);
  set_tensor(osc_omega, 1);
  set_tensor(osc_amp, 2);
  set_tensor(carrier_real, 3);
  set_tensor(carrier_imag, 4);
  set_tensor(carrier_omega, 5);
  set_tensor(carrier_gate_width, 6);
  set_tensor(carrier_state, 7);
  set_tensor(carrier_anchor_idx, 8);
  set_tensor(carrier_anchor_phase, 9);
  set_tensor(carrier_anchor_weight, 10);
  set_tensor(energy_stats, 11);

  uint32_t num_carriers_u = (uint32_t)num_carriers_i;
  [encoder setBytes:&num_carriers_u length:sizeof(uint32_t) atIndex:12];

  // Reuse SpectralCarrierParams for dt/coupling/gate bounds; other fields unused here.
  SpectralCarrierParams prm;
  prm.num_osc = (uint32_t)N;
  prm.max_carriers = (uint32_t)max_carriers;
  prm.num_carriers = (uint32_t)num_carriers_i;
  prm.dt = (float)dt;
  prm.coupling_scale = (float)coupling_scale;
  prm.carrier_reg = 0.0f;
  prm.temperature = (float)temperature;
  prm.rng_seed = (uint32_t)rng_seed;
  prm.conflict_threshold = 0.0f;
  prm.offender_weight_floor = 0.0f;
  prm.gate_width_min = (float)gate_width_min;
  prm.gate_width_max = (float)gate_width_max;
  prm.ema_alpha = 0.0f;
  prm.recenter_alpha = 0.0f;
  prm.mode = 0u;
  prm.anchor_random_eps = 0.0f;
  prm.stable_amp_threshold = 0.0f;
  prm.crystallize_amp_threshold = 0.0f;
  prm.crystallize_conflict_threshold = 0.0f;
  prm.crystallize_age = 1u;
  prm.crystallized_coupling_boost = (float)crystallized_coupling_boost;
  prm.volatile_decay_mul = 1.0f;
  prm.stable_decay_mul = 1.0f;
  prm.crystallized_decay_mul = 1.0f;
  prm.topdown_phase_scale = (float)topdown_phase_scale;
  prm.topdown_energy_scale = 0.0f;
  prm.topdown_random_energy_eps = 0.0f;
  prm.repulsion_scale = 0.0f;
  [encoder setBytes:&prm length:sizeof(SpectralCarrierParams) atIndex:13];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const NSUInteger num_groups = (N + kThreadsPerThreadgroup - 1) / kThreadsPerThreadgroup;
  const MTLSize grid = MTLSizeMake(num_groups, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  stream->endKernelCoalescing();
}

void spectral_topdown_bias_energies(
    at::Tensor osc_energy,          // (N,) fp32 MPS in/out
    at::Tensor osc_amp,             // (N,) fp32 MPS
    at::Tensor carrier_state,       // (maxM,) int32 MPS
    at::Tensor carrier_anchor_idx,  // (maxM*anchors,) int32 MPS
    at::Tensor carrier_anchor_weight, // (maxM*anchors,) fp32 MPS
    int64_t num_carriers_i,
    int64_t max_carriers,
    double dt,
    uint32_t rng_seed,
    int64_t mode,
    double topdown_energy_scale,
    double topdown_random_energy_eps) {

  check_contig_1d(osc_energy, "spectral_topdown_bias: osc_energy");
  check_contig_1d(osc_amp, "spectral_topdown_bias: osc_amp");
  check_contig_1d(carrier_state, "spectral_topdown_bias: carrier_state");
  check_contig_1d(carrier_anchor_idx, "spectral_topdown_bias: carrier_anchor_idx");
  check_contig_1d(carrier_anchor_weight, "spectral_topdown_bias: carrier_anchor_weight");
  TORCH_CHECK(osc_energy.device().is_mps(), "spectral_topdown_bias: osc_energy must be on MPS");
  TORCH_CHECK(osc_amp.device().is_mps(), "spectral_topdown_bias: osc_amp must be on MPS");
  TORCH_CHECK(osc_energy.dtype() == at::kFloat, "spectral_topdown_bias: osc_energy must be fp32");
  TORCH_CHECK(osc_amp.dtype() == at::kFloat, "spectral_topdown_bias: osc_amp must be fp32");
  TORCH_CHECK(carrier_state.dtype() == at::kInt, "spectral_topdown_bias: carrier_state must be int32");
  TORCH_CHECK(carrier_anchor_idx.dtype() == at::kInt, "spectral_topdown_bias: carrier_anchor_idx must be int32");

  const int64_t N = osc_energy.size(0);
  if (N == 0 || num_carriers_i == 0) return;

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();

  id<MTLComputePipelineState> pipeline =
      ensure_pipeline(device, &g_pipeline_spectral_topdown_bias_energies, "spectral_topdown_bias_energies");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(osc_energy, 0);
  set_tensor(osc_amp, 1);
  set_tensor(carrier_state, 2);
  set_tensor(carrier_anchor_idx, 3);
  set_tensor(carrier_anchor_weight, 4);

  uint32_t num_carriers_u = (uint32_t)num_carriers_i;
  [encoder setBytes:&num_carriers_u length:sizeof(uint32_t) atIndex:5];

  SpectralCarrierParams prm;
  prm.num_osc = (uint32_t)N;
  prm.max_carriers = (uint32_t)max_carriers;
  prm.num_carriers = (uint32_t)num_carriers_i;
  prm.dt = (float)dt;
  prm.coupling_scale = 0.0f;
  prm.carrier_reg = 0.0f;
  prm.temperature = 0.0f;
  prm.rng_seed = (uint32_t)rng_seed;
  prm.conflict_threshold = 0.0f;
  prm.offender_weight_floor = 0.0f;
  prm.gate_width_min = 0.0f;
  prm.gate_width_max = 0.0f;
  prm.ema_alpha = 0.0f;
  prm.recenter_alpha = 0.0f;
  prm.mode = (uint32_t)mode;
  prm.anchor_random_eps = 0.0f;
  prm.stable_amp_threshold = 0.0f;
  prm.crystallize_amp_threshold = 0.0f;
  prm.crystallize_conflict_threshold = 0.0f;
  prm.crystallize_age = 1u;
  prm.crystallized_coupling_boost = 0.0f;
  prm.volatile_decay_mul = 1.0f;
  prm.stable_decay_mul = 1.0f;
  prm.crystallized_decay_mul = 1.0f;
  prm.topdown_phase_scale = 0.0f;
  prm.topdown_energy_scale = (float)topdown_energy_scale;
  prm.topdown_random_energy_eps = (float)topdown_random_energy_eps;
  prm.repulsion_scale = 0.0f;
  [encoder setBytes:&prm length:sizeof(SpectralCarrierParams) atIndex:6];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const NSUInteger num_groups = (num_carriers_i + kThreadsPerThreadgroup - 1) / kThreadsPerThreadgroup;
  const MTLSize grid = MTLSizeMake(num_groups, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  stream->endKernelCoalescing();
}

int64_t spectral_spawn_uncoupled(
    at::Tensor osc_phase,           // (N,) fp32 MPS
    at::Tensor osc_omega,           // (N,) fp32 MPS
    at::Tensor osc_amp,             // (N,) fp32 MPS
    at::Tensor carrier_omega,       // (maxM,) fp32 MPS (read)
    at::Tensor carrier_gate_width,  // (maxM,) fp32 MPS (read)
    at::Tensor carrier_real,        // (maxM,) fp32 MPS (write for new)
    at::Tensor carrier_imag,        // (maxM,) fp32 MPS (write for new)
    at::Tensor carrier_omega_out,   // (maxM,) fp32 MPS (write for new)
    at::Tensor carrier_gate_width_out, // (maxM,) fp32 MPS (write for new)
    at::Tensor carrier_conflict,    // (maxM,) fp32 MPS (write for new)
    at::Tensor carrier_state,       // (maxM,) int32 MPS (write for new)
    at::Tensor carrier_age,         // (maxM,) int32 MPS (write for new)
    at::Tensor carrier_anchor_idx,  // (maxM*anchors,) int32 MPS (write for new)
    at::Tensor carrier_anchor_phase,// (maxM*anchors,) fp32 MPS (write for new)
    at::Tensor carrier_anchor_weight,// (maxM*anchors,) fp32 MPS (write for new)
    at::Tensor num_carriers_buf,    // (1,) int32 MPS (atomic counter)
    int64_t num_carriers,
    int64_t max_carriers,
    double coupling_threshold,
    double gate_width_init,
    double gate_width_min,
    double gate_width_max) {

  check_contig_1d(osc_phase, "spectral_spawn_uncoupled: osc_phase");
  check_contig_1d(osc_omega, "spectral_spawn_uncoupled: osc_omega");
  check_contig_1d(osc_amp, "spectral_spawn_uncoupled: osc_amp");
  check_contig_1d(carrier_state, "spectral_spawn_uncoupled: carrier_state");
  check_contig_1d(carrier_age, "spectral_spawn_uncoupled: carrier_age");
  check_contig_1d(carrier_anchor_idx, "spectral_spawn_uncoupled: carrier_anchor_idx");
  check_contig_1d(carrier_anchor_phase, "spectral_spawn_uncoupled: carrier_anchor_phase");
  check_contig_1d(carrier_anchor_weight, "spectral_spawn_uncoupled: carrier_anchor_weight");
  TORCH_CHECK(carrier_state.dtype() == at::kInt, "spectral_spawn_uncoupled: carrier_state must be int32");
  TORCH_CHECK(carrier_age.dtype() == at::kInt, "spectral_spawn_uncoupled: carrier_age must be int32");
  TORCH_CHECK(carrier_anchor_idx.dtype() == at::kInt, "spectral_spawn_uncoupled: carrier_anchor_idx must be int32");

  const int64_t N = osc_phase.size(0);
  if (N == 0) return num_carriers;

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();

  id<MTLComputePipelineState> pipeline =
      ensure_pipeline(device, &g_pipeline_spectral_spawn_uncoupled, "spectral_spawn_uncoupled");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(osc_phase, 0);
  set_tensor(osc_omega, 1);
  set_tensor(osc_amp, 2);
  set_tensor(carrier_omega, 3);
  set_tensor(carrier_gate_width, 4);
  set_tensor(carrier_real, 5);
  set_tensor(carrier_imag, 6);
  set_tensor(carrier_omega_out, 7);
  set_tensor(carrier_gate_width_out, 8);
  set_tensor(carrier_conflict, 9);
  set_tensor(carrier_state, 10);
  set_tensor(carrier_age, 11);
  set_tensor(carrier_anchor_idx, 12);
  set_tensor(carrier_anchor_phase, 13);
  set_tensor(carrier_anchor_weight, 14);
  set_tensor(num_carriers_buf, 15);

  uint32_t num_carriers_u = (uint32_t)num_carriers;
  uint32_t max_carriers_u = (uint32_t)max_carriers;
  float coupling_threshold_f = (float)coupling_threshold;
  float gate_width_init_f = (float)gate_width_init;
  float gate_width_min_f = (float)gate_width_min;
  float gate_width_max_f = (float)gate_width_max;
  uint32_t num_osc_u = (uint32_t)N;

  [encoder setBytes:&num_carriers_u length:sizeof(uint32_t) atIndex:16];
  [encoder setBytes:&max_carriers_u length:sizeof(uint32_t) atIndex:17];
  [encoder setBytes:&coupling_threshold_f length:sizeof(float) atIndex:18];
  [encoder setBytes:&gate_width_init_f length:sizeof(float) atIndex:19];
  [encoder setBytes:&gate_width_min_f length:sizeof(float) atIndex:20];
  [encoder setBytes:&gate_width_max_f length:sizeof(float) atIndex:21];
  [encoder setBytes:&num_osc_u length:sizeof(uint32_t) atIndex:22];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const NSUInteger num_groups = (N + kThreadsPerThreadgroup - 1) / kThreadsPerThreadgroup;
  const MTLSize grid = MTLSizeMake(num_groups, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  stream->endKernelCoalescing();

  // Return updated carrier count (requires sync)
  int64_t new_count = num_carriers_buf.to(at::kCPU).item<int32_t>();
  return std::min(new_count, max_carriers);
}

// ============================================================================
// Particle Generation Kernels
// ============================================================================

void generate_particles(
    at::Tensor positions,      // (N, 3) fp32 MPS out
    at::Tensor velocities,     // (N, 3) fp32 MPS out
    at::Tensor energies,       // (N,) fp32 MPS out
    at::Tensor heats,          // (N,) fp32 MPS out
    at::Tensor excitations,    // (N,) fp32 MPS out
    at::Tensor masses,         // (N,) fp32 MPS out
    at::Tensor random_pos,     // (N, 3) fp32 MPS (uniform [0,1])
    at::Tensor random_props,   // (N, 4) fp32 MPS (uniform [0,1])
    int64_t pattern,           // 0=cluster, 1=line, 2=sphere, 3=random, 4=grid
    double grid_x,
    double grid_y,
    double grid_z,
    double energy_scale,
    double center_x,
    double center_y,
    double center_z,
    double spread,
    double dir_x,
    double dir_y,
    double dir_z) {
  
  const int64_t N = positions.size(0);
  
  TORCH_CHECK(positions.device().is_mps(), "generate_particles: positions must be on MPS");
  TORCH_CHECK(positions.is_contiguous() && positions.dim() == 2 && positions.size(1) == 3,
              "generate_particles: positions must be contiguous (N, 3)");
  
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  
  // Step 1: Generate positions
  {
    stream->endKernelCoalescing();
    id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    
    id<MTLComputePipelineState> pipeline = ensure_pipeline(device, &g_pipeline_generate_particle_positions, "generate_particle_positions");
    [encoder setComputePipelineState:pipeline];
    
    auto set_tensor = [&](const at::Tensor& t, int idx) {
      id<MTLBuffer> buf = storage_as_mtlbuffer(t);
      const NSUInteger off = storage_offset_bytes(t);
      [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
    };
    
    set_tensor(positions, 0);
    set_tensor(random_pos, 1);
    
    ParticleGenParams prm;
    prm.num_particles = (uint32_t)N;
    prm.grid_x = (float)grid_x;
    prm.grid_y = (float)grid_y;
    prm.grid_z = (float)grid_z;
    prm.energy_scale = (float)energy_scale;
    prm.pattern = (uint32_t)pattern;
    prm.center_x = (float)center_x;
    prm.center_y = (float)center_y;
    prm.center_z = (float)center_z;
    prm.spread = (float)spread;
    prm.dir_x = (float)dir_x;
    prm.dir_y = (float)dir_y;
    prm.dir_z = (float)dir_z;
    [encoder setBytes:&prm length:sizeof(ParticleGenParams) atIndex:2];
    
    const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
    const NSUInteger num_groups = (N + kThreadsPerThreadgroup - 1) / kThreadsPerThreadgroup;
    const MTLSize grid = MTLSizeMake(num_groups, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    
    stream->endKernelCoalescing();
  }
  
  // Need to sync to compute mean position
  stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
  
  // Compute center from positions
  at::Tensor pos_mean = positions.mean(0);
  // NOTE: avoid `.item()` directly on MPS tensors inside extensions.
  at::Tensor pos_mean_cpu = pos_mean.to(at::kCPU);
  float mean_x = pos_mean_cpu[0].item<float>();
  float mean_y = pos_mean_cpu[1].item<float>();
  float mean_z = pos_mean_cpu[2].item<float>();
  
  // Step 2: Initialize properties
  {
    stream->endKernelCoalescing();
    id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    
    id<MTLComputePipelineState> pipeline = ensure_pipeline(device, &g_pipeline_initialize_particle_properties, "initialize_particle_properties");
    [encoder setComputePipelineState:pipeline];
    
    auto set_tensor = [&](const at::Tensor& t, int idx) {
      id<MTLBuffer> buf = storage_as_mtlbuffer(t);
      const NSUInteger off = storage_offset_bytes(t);
      [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
    };
    
    set_tensor(positions, 0);
    set_tensor(velocities, 1);
    set_tensor(energies, 2);
    set_tensor(heats, 3);
    set_tensor(excitations, 4);
    set_tensor(masses, 5);
    set_tensor(random_props, 6);
    
    ParticleGenParams prm;
    prm.num_particles = (uint32_t)N;
    prm.grid_x = (float)grid_x;
    prm.grid_y = (float)grid_y;
    prm.grid_z = (float)grid_z;
    prm.energy_scale = (float)energy_scale;
    prm.pattern = (uint32_t)pattern;
    prm.center_x = (float)center_x;
    prm.center_y = (float)center_y;
    prm.center_z = (float)center_z;
    prm.spread = (float)spread;
    [encoder setBytes:&prm length:sizeof(ParticleGenParams) atIndex:7];
    
    [encoder setBytes:&mean_x length:sizeof(float) atIndex:8];
    [encoder setBytes:&mean_y length:sizeof(float) atIndex:9];
    [encoder setBytes:&mean_z length:sizeof(float) atIndex:10];
    
    const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
    const NSUInteger num_groups = (N + kThreadsPerThreadgroup - 1) / kThreadsPerThreadgroup;
    const MTLSize grid = MTLSizeMake(num_groups, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    
    stream->endKernelCoalescing();
  }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Manifold physics kernels
  m.def("manifold_scatter_particles", &manifold_scatter_particles, "Scatter particles to gravity/heat fields (Metal/MPS, fp32)");
  m.def("manifold_gather_update_particles", &manifold_gather_update_particles, "Gather fields and update particles (Metal/MPS, fp32)");
  m.def("manifold_diffuse_heat", &manifold_diffuse_heat, "Diffuse heat field (Metal/MPS, fp32)");
  m.def("manifold_poisson_step", &manifold_poisson_step, "Poisson Jacobi step for gravity (Metal/MPS, fp32)");
  m.def("manifold_clear_field", &manifold_clear_field, "Clear a field to zero (Metal/MPS, fp32)");
  // Adaptive thermodynamics (global statistics, GPU-only)
  m.def("thermo_reduce_energy_stats", &thermo_reduce_energy_stats, "Reduce energy stats: [mean_abs, mean, std, count] (Metal/MPS, fp32)");
  m.def("particle_interactions", &particle_interactions, "Particle collision + excitation transfer O(N²) (Metal/MPS, fp32)");
  // Tiled scatter (reduced atomic contention)
  m.def("scatter_particle_tiled", &scatter_particle_tiled, "Tiled scatter with threadgroup reduction (Metal/MPS, fp32)");
  // Spatial hash grid (O(N) collisions)
  m.def("spatial_hash_assign", &spatial_hash_assign, "Assign particles to spatial hash cells (Metal/MPS)");
  m.def("spatial_hash_prefix_sum", &spatial_hash_prefix_sum, "Compute prefix sum of cell counts (Metal/MPS)");
  m.def("spatial_hash_scatter", &spatial_hash_scatter, "Scatter particle indices to sorted array (Metal/MPS)");
  m.def("spatial_hash_collisions", &spatial_hash_collisions, "Particle collisions using spatial hash O(N) (Metal/MPS, fp32)");
  // Spectral (resonance potential) carrier coupling kernels
  m.def("spectral_carrier_update_and_split", &spectral_carrier_update_and_split, "Spectral carrier update + conflict split (Metal/MPS, fp32)");
  m.def("spectral_update_oscillator_phases", &spectral_update_oscillator_phases, "Spectral oscillator phase update from carriers (Metal/MPS, fp32)");
  m.def("spectral_topdown_bias_energies", &spectral_topdown_bias_energies, "Top-down energy bias from crystallized carriers (Metal/MPS, fp32)");
  m.def("spectral_spawn_uncoupled", &spectral_spawn_uncoupled, "Spawn carriers for uncoupled oscillators (Metal/MPS, fp32)");
  // Particle generation kernels
  m.def("generate_particles", &generate_particles, "Generate particles with pattern (Metal/MPS, fp32)");
}
