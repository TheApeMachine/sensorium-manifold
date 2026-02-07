#include <torch/extension.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#include <dlfcn.h>
#include <filesystem>
#include <mutex>
#include <string>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace fs = std::filesystem;

namespace {

struct HaloPackParams {
  uint32_t gx;
  uint32_t gy;
  uint32_t gz;
  uint32_t halo;
  uint32_t start;
};

struct MigrationBounds {
  float lo_x;
  float lo_y;
  float lo_z;
  float hi_x;
  float hi_y;
  float hi_z;
  uint32_t n;
};

struct JacobiHaloParams {
  uint32_t gx;
  uint32_t gy;
  uint32_t gz;
  float dx;
};

struct AdvanceInteriorParams {
  uint32_t gx;
  uint32_t gy;
  uint32_t gz;
  uint32_t h;
  float dt;
  float dx;
  float gamma;
  float rho_min;
  float viscosity;
  float thermal_diff;
};

constexpr NSUInteger kThreadsPerThreadgroup = 256;

static id<MTLLibrary> g_lib = nil;
static id<MTLComputePipelineState> g_pipeline_pack_x = nil;
static id<MTLComputePipelineState> g_pipeline_pack_y = nil;
static id<MTLComputePipelineState> g_pipeline_pack_z = nil;
static id<MTLComputePipelineState> g_pipeline_unpack_x = nil;
static id<MTLComputePipelineState> g_pipeline_unpack_y = nil;
static id<MTLComputePipelineState> g_pipeline_unpack_z = nil;
static id<MTLComputePipelineState> g_pipeline_classify = nil;
static id<MTLComputePipelineState> g_pipeline_jacobi_halo = nil;
static id<MTLComputePipelineState> g_pipeline_advance_interior_halo = nil;
static std::mutex g_pipeline_mutex;

static std::string metallib_path_for_this_module() {
  Dl_info info;
  if (dladdr((void*)&metallib_path_for_this_module, &info) == 0 || info.dli_fname == nullptr) {
    return std::string();
  }
  fs::path so_path(info.dli_fname);
  fs::path lib_path = so_path.parent_path() / "distributed_ops.metallib";
  return lib_path.string();
}

static inline id<MTLBuffer> storage_as_mtlbuffer(const at::Tensor& t) {
  const auto& dp = t.storage().data_ptr();
  void* ctx = dp.get_context();
  TORCH_CHECK(ctx != nullptr, "distributed_metal_ops: missing MTLBuffer context");
  return (__bridge id<MTLBuffer>)ctx;
}

static inline NSUInteger storage_offset_bytes(const at::Tensor& t) {
  return (NSUInteger)(t.storage_offset() * (int64_t)t.element_size());
}

static void ensure_library_locked(id<MTLDevice> device) {
  if (g_lib != nil) return;
  const std::string lib_path = metallib_path_for_this_module();
  TORCH_CHECK(!lib_path.empty(), "distributed_metal_ops: failed to locate extension path");
  NSString* ns_path = [NSString stringWithUTF8String:lib_path.c_str()];
  NSURL* url = [NSURL fileURLWithPath:ns_path];
  NSError* err = nil;
  g_lib = [device newLibraryWithURL:url error:&err];
  if (g_lib == nil) {
    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
    TORCH_CHECK(false, "distributed_metal_ops: failed to load metallib at ", lib_path, ": ", msg);
  }
}

static id<MTLComputePipelineState> ensure_pipeline(
    id<MTLDevice> device,
    id<MTLComputePipelineState> __strong* pipeline,
    const char* fn_name) {
  std::lock_guard<std::mutex> lock(g_pipeline_mutex);
  ensure_library_locked(device);
  if (*pipeline != nil) {
    return *pipeline;
  }
  NSString* ns_fn = [NSString stringWithUTF8String:fn_name];
  id<MTLFunction> fn = [g_lib newFunctionWithName:ns_fn];
  TORCH_CHECK(fn != nil, "distributed_metal_ops: function `", fn_name, "` not found");
  NSError* err = nil;
  *pipeline = [device newComputePipelineStateWithFunction:fn error:&err];
  if (*pipeline == nil) {
    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
    TORCH_CHECK(false, "distributed_metal_ops: failed to create pipeline: ", msg);
  }
  return *pipeline;
}

struct ComputeKernel {
  id<MTLDevice> device;
  at::mps::MPSStream* stream;
  id<MTLComputeCommandEncoder> encoder;
  id<MTLComputePipelineState> pipeline;

  ComputeKernel(id<MTLComputePipelineState> __strong* cache, const char* fn_name) {
    device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
    stream = at::mps::getCurrentMPSStream();
    stream->endKernelCoalescing();
    encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    pipeline = ensure_pipeline(device, cache, fn_name);
    [encoder setComputePipelineState:pipeline];
  }

  inline void set_tensor(const at::Tensor& t, int idx) {
    [encoder setBuffer:storage_as_mtlbuffer(t) offset:storage_offset_bytes(t) atIndex:(NSUInteger)idx];
  }

  template <class T>
  inline void set_bytes(const T& v, int idx) {
    [encoder setBytes:&v length:(NSUInteger)sizeof(T) atIndex:(NSUInteger)idx];
  }

  inline void dispatch_1d(int64_t n_threads) {
    const NSUInteger tg = (NSUInteger)kThreadsPerThreadgroup;
    const NSUInteger groups = (n_threads + (int64_t)tg - 1) / (int64_t)tg;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  }

  ~ComputeKernel() { stream->endKernelCoalescing(); }
};

static void check_mps_contig(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.dtype() == at::kFloat || t.dtype() == at::kInt, name, " dtype must be fp32 or int32");
}

void distributed_pack_halo_x(
    at::Tensor field,
    at::Tensor out_face,
    int64_t halo,
    int64_t start_x) {
  check_mps_contig(field, "distributed_pack_halo_x: field");
  check_mps_contig(out_face, "distributed_pack_halo_x: out_face");
  TORCH_CHECK(field.dim() == 3, "distributed_pack_halo_x: field must be 3D");
  TORCH_CHECK(out_face.dim() == 3, "distributed_pack_halo_x: out_face must be 3D");

  HaloPackParams p;
  p.gx = (uint32_t)field.size(0);
  p.gy = (uint32_t)field.size(1);
  p.gz = (uint32_t)field.size(2);
  p.halo = (uint32_t)halo;
  p.start = (uint32_t)start_x;
  ComputeKernel k(&g_pipeline_pack_x, "distributed_pack_x_face");
  k.set_tensor(field, 0);
  k.set_tensor(out_face, 1);
  k.set_bytes(p, 2);
  k.dispatch_1d(out_face.numel());
}

void distributed_pack_halo_y(
    at::Tensor field,
    at::Tensor out_face,
    int64_t halo,
    int64_t start_y) {
  check_mps_contig(field, "distributed_pack_halo_y: field");
  check_mps_contig(out_face, "distributed_pack_halo_y: out_face");
  TORCH_CHECK(field.dim() == 3, "distributed_pack_halo_y: field must be 3D");
  TORCH_CHECK(out_face.dim() == 3, "distributed_pack_halo_y: out_face must be 3D");

  HaloPackParams p;
  p.gx = (uint32_t)field.size(0);
  p.gy = (uint32_t)field.size(1);
  p.gz = (uint32_t)field.size(2);
  p.halo = (uint32_t)halo;
  p.start = (uint32_t)start_y;
  ComputeKernel k(&g_pipeline_pack_y, "distributed_pack_y_face");
  k.set_tensor(field, 0);
  k.set_tensor(out_face, 1);
  k.set_bytes(p, 2);
  k.dispatch_1d(out_face.numel());
}

void distributed_pack_halo_z(
    at::Tensor field,
    at::Tensor out_face,
    int64_t halo,
    int64_t start_z) {
  check_mps_contig(field, "distributed_pack_halo_z: field");
  check_mps_contig(out_face, "distributed_pack_halo_z: out_face");
  TORCH_CHECK(field.dim() == 3, "distributed_pack_halo_z: field must be 3D");
  TORCH_CHECK(out_face.dim() == 3, "distributed_pack_halo_z: out_face must be 3D");

  HaloPackParams p;
  p.gx = (uint32_t)field.size(0);
  p.gy = (uint32_t)field.size(1);
  p.gz = (uint32_t)field.size(2);
  p.halo = (uint32_t)halo;
  p.start = (uint32_t)start_z;
  ComputeKernel k(&g_pipeline_pack_z, "distributed_pack_z_face");
  k.set_tensor(field, 0);
  k.set_tensor(out_face, 1);
  k.set_bytes(p, 2);
  k.dispatch_1d(out_face.numel());
}

void distributed_classify_faces(
    at::Tensor positions,
    at::Tensor out_codes,
    double lo_x,
    double lo_y,
    double lo_z,
    double hi_x,
    double hi_y,
    double hi_z) {
  check_mps_contig(positions, "distributed_classify_faces: positions");
  check_mps_contig(out_codes, "distributed_classify_faces: out_codes");
  TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3,
              "distributed_classify_faces: positions must be (N,3)");
  TORCH_CHECK(out_codes.dim() == 1, "distributed_classify_faces: out_codes must be 1D");

  MigrationBounds b;
  b.lo_x = (float)lo_x;
  b.lo_y = (float)lo_y;
  b.lo_z = (float)lo_z;
  b.hi_x = (float)hi_x;
  b.hi_y = (float)hi_y;
  b.hi_z = (float)hi_z;
  b.n = (uint32_t)positions.size(0);

  ComputeKernel k(&g_pipeline_classify, "distributed_classify_faces");
  k.set_tensor(positions, 0);
  k.set_tensor(out_codes, 1);
  k.set_bytes(b, 2);
  k.dispatch_1d(positions.size(0));
}

void distributed_unpack_halo_x(
    at::Tensor in_face,
    at::Tensor field,
    int64_t halo,
    int64_t start_x) {
  check_mps_contig(in_face, "distributed_unpack_halo_x: in_face");
  check_mps_contig(field, "distributed_unpack_halo_x: field");
  TORCH_CHECK(field.dim() == 3, "distributed_unpack_halo_x: field must be 3D");

  HaloPackParams p;
  p.gx = (uint32_t)field.size(0);
  p.gy = (uint32_t)field.size(1);
  p.gz = (uint32_t)field.size(2);
  p.halo = (uint32_t)halo;
  p.start = (uint32_t)start_x;
  ComputeKernel k(&g_pipeline_unpack_x, "distributed_unpack_x_face");
  k.set_tensor(in_face, 0);
  k.set_tensor(field, 1);
  k.set_bytes(p, 2);
  k.dispatch_1d(in_face.numel());
}

void distributed_unpack_halo_y(
    at::Tensor in_face,
    at::Tensor field,
    int64_t halo,
    int64_t start_y) {
  check_mps_contig(in_face, "distributed_unpack_halo_y: in_face");
  check_mps_contig(field, "distributed_unpack_halo_y: field");
  TORCH_CHECK(field.dim() == 3, "distributed_unpack_halo_y: field must be 3D");

  HaloPackParams p;
  p.gx = (uint32_t)field.size(0);
  p.gy = (uint32_t)field.size(1);
  p.gz = (uint32_t)field.size(2);
  p.halo = (uint32_t)halo;
  p.start = (uint32_t)start_y;
  ComputeKernel k(&g_pipeline_unpack_y, "distributed_unpack_y_face");
  k.set_tensor(in_face, 0);
  k.set_tensor(field, 1);
  k.set_bytes(p, 2);
  k.dispatch_1d(in_face.numel());
}

void distributed_unpack_halo_z(
    at::Tensor in_face,
    at::Tensor field,
    int64_t halo,
    int64_t start_z) {
  check_mps_contig(in_face, "distributed_unpack_halo_z: in_face");
  check_mps_contig(field, "distributed_unpack_halo_z: field");
  TORCH_CHECK(field.dim() == 3, "distributed_unpack_halo_z: field must be 3D");

  HaloPackParams p;
  p.gx = (uint32_t)field.size(0);
  p.gy = (uint32_t)field.size(1);
  p.gz = (uint32_t)field.size(2);
  p.halo = (uint32_t)halo;
  p.start = (uint32_t)start_z;
  ComputeKernel k(&g_pipeline_unpack_z, "distributed_unpack_z_face");
  k.set_tensor(in_face, 0);
  k.set_tensor(field, 1);
  k.set_bytes(p, 2);
  k.dispatch_1d(in_face.numel());
}

void distributed_jacobi_halo(
    at::Tensor phi,
    at::Tensor rhs,
    at::Tensor halo_xm,
    at::Tensor halo_xp,
    at::Tensor halo_ym,
    at::Tensor halo_yp,
    at::Tensor halo_zm,
    at::Tensor halo_zp,
    at::Tensor out_phi,
    double dx) {
  check_mps_contig(phi, "distributed_jacobi_halo: phi");
  check_mps_contig(rhs, "distributed_jacobi_halo: rhs");
  check_mps_contig(halo_xm, "distributed_jacobi_halo: halo_xm");
  check_mps_contig(halo_xp, "distributed_jacobi_halo: halo_xp");
  check_mps_contig(halo_ym, "distributed_jacobi_halo: halo_ym");
  check_mps_contig(halo_yp, "distributed_jacobi_halo: halo_yp");
  check_mps_contig(halo_zm, "distributed_jacobi_halo: halo_zm");
  check_mps_contig(halo_zp, "distributed_jacobi_halo: halo_zp");
  check_mps_contig(out_phi, "distributed_jacobi_halo: out_phi");
  TORCH_CHECK(phi.dim() == 3 && rhs.dim() == 3 && out_phi.dim() == 3,
              "distributed_jacobi_halo: phi/rhs/out must be 3D");

  JacobiHaloParams p;
  p.gx = (uint32_t)phi.size(0);
  p.gy = (uint32_t)phi.size(1);
  p.gz = (uint32_t)phi.size(2);
  p.dx = (float)dx;

  ComputeKernel k(&g_pipeline_jacobi_halo, "distributed_jacobi_halo");
  k.set_tensor(phi, 0);
  k.set_tensor(rhs, 1);
  k.set_tensor(halo_xm, 2);
  k.set_tensor(halo_xp, 3);
  k.set_tensor(halo_ym, 4);
  k.set_tensor(halo_yp, 5);
  k.set_tensor(halo_zm, 6);
  k.set_tensor(halo_zp, 7);
  k.set_tensor(out_phi, 8);
  k.set_bytes(p, 9);
  k.dispatch_1d(phi.numel());
}

void distributed_advance_interior_halo(
    at::Tensor rho_ext,
    at::Tensor mx_ext,
    at::Tensor my_ext,
    at::Tensor mz_ext,
    at::Tensor e_ext,
    at::Tensor phi_ext,
    at::Tensor out_rho,
    at::Tensor out_mx,
    at::Tensor out_my,
    at::Tensor out_mz,
    at::Tensor out_e,
    int64_t h,
    double dt,
    double dx,
    double gamma,
    double rho_min,
    double viscosity,
    double thermal_diff) {
  check_mps_contig(rho_ext, "distributed_advance_interior_halo: rho_ext");
  check_mps_contig(mx_ext, "distributed_advance_interior_halo: mx_ext");
  check_mps_contig(my_ext, "distributed_advance_interior_halo: my_ext");
  check_mps_contig(mz_ext, "distributed_advance_interior_halo: mz_ext");
  check_mps_contig(e_ext, "distributed_advance_interior_halo: e_ext");
  check_mps_contig(phi_ext, "distributed_advance_interior_halo: phi_ext");
  check_mps_contig(out_rho, "distributed_advance_interior_halo: out_rho");
  check_mps_contig(out_mx, "distributed_advance_interior_halo: out_mx");
  check_mps_contig(out_my, "distributed_advance_interior_halo: out_my");
  check_mps_contig(out_mz, "distributed_advance_interior_halo: out_mz");
  check_mps_contig(out_e, "distributed_advance_interior_halo: out_e");

  AdvanceInteriorParams p;
  p.gx = (uint32_t)out_rho.size(0);
  p.gy = (uint32_t)out_rho.size(1);
  p.gz = (uint32_t)out_rho.size(2);
  p.h = (uint32_t)h;
  p.dt = (float)dt;
  p.dx = (float)dx;
  p.gamma = (float)gamma;
  p.rho_min = (float)rho_min;
  p.viscosity = (float)viscosity;
  p.thermal_diff = (float)thermal_diff;

  ComputeKernel k(&g_pipeline_advance_interior_halo, "distributed_advance_interior_halo");
  k.set_tensor(rho_ext, 0);
  k.set_tensor(mx_ext, 1);
  k.set_tensor(my_ext, 2);
  k.set_tensor(mz_ext, 3);
  k.set_tensor(e_ext, 4);
  k.set_tensor(phi_ext, 5);
  k.set_tensor(out_rho, 6);
  k.set_tensor(out_mx, 7);
  k.set_tensor(out_my, 8);
  k.set_tensor(out_mz, 9);
  k.set_tensor(out_e, 10);
  k.set_bytes(p, 11);
  k.dispatch_1d(out_rho.numel());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("distributed_pack_halo_x", &distributed_pack_halo_x, "Pack X face halo");
  m.def("distributed_pack_halo_y", &distributed_pack_halo_y, "Pack Y face halo");
  m.def("distributed_pack_halo_z", &distributed_pack_halo_z, "Pack Z face halo");
  m.def("distributed_unpack_halo_x", &distributed_unpack_halo_x, "Unpack X face halo");
  m.def("distributed_unpack_halo_y", &distributed_unpack_halo_y, "Unpack Y face halo");
  m.def("distributed_unpack_halo_z", &distributed_unpack_halo_z, "Unpack Z face halo");
  m.def("distributed_classify_faces", &distributed_classify_faces, "Classify particle migration faces");
  m.def("distributed_jacobi_halo", &distributed_jacobi_halo, "Jacobi step with face halos");
  m.def("distributed_advance_interior_halo", &distributed_advance_interior_halo, "Advance interior state with halo-aware stencils");
}
