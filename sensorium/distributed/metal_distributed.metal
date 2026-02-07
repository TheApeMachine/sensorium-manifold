#include <metal_stdlib>
using namespace metal;

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

kernel void distributed_pack_x_face(
    device const float* field [[buffer(0)]],
    device float* out_face [[buffer(1)]],
    constant HaloPackParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = p.halo * p.gy * p.gz;
    if (gid >= n) return;
    uint yz = p.gy * p.gz;
    uint hx = gid / yz;
    uint rem = gid - hx * yz;
    uint y = rem / p.gz;
    uint z = rem - y * p.gz;
    uint x = p.start + hx;
    uint src = x * yz + y * p.gz + z;
    out_face[gid] = field[src];
}

kernel void distributed_pack_y_face(
    device const float* field [[buffer(0)]],
    device float* out_face [[buffer(1)]],
    constant HaloPackParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = p.gx * p.halo * p.gz;
    if (gid >= n) return;
    uint x = gid / (p.halo * p.gz);
    uint rem = gid - x * (p.halo * p.gz);
    uint hy = rem / p.gz;
    uint z = rem - hy * p.gz;
    uint y = p.start + hy;
    uint yz = p.gy * p.gz;
    uint src = x * yz + y * p.gz + z;
    out_face[gid] = field[src];
}

kernel void distributed_pack_z_face(
    device const float* field [[buffer(0)]],
    device float* out_face [[buffer(1)]],
    constant HaloPackParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = p.gx * p.gy * p.halo;
    if (gid >= n) return;
    uint x = gid / (p.gy * p.halo);
    uint rem = gid - x * (p.gy * p.halo);
    uint y = rem / p.halo;
    uint hz = rem - y * p.halo;
    uint z = p.start + hz;
    uint yz = p.gy * p.gz;
    uint src = x * yz + y * p.gz + z;
    out_face[gid] = field[src];
}

kernel void distributed_classify_faces(
    device const float* positions [[buffer(0)]],
    device int* codes [[buffer(1)]],
    constant MigrationBounds& b [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= b.n) return;
    float x = positions[gid * 3 + 0];
    float y = positions[gid * 3 + 1];
    float z = positions[gid * 3 + 2];
    int code = 0;
    if (x < b.lo_x) code = 1;
    else if (x >= b.hi_x) code = 2;
    else if (y < b.lo_y) code = 3;
    else if (y >= b.hi_y) code = 4;
    else if (z < b.lo_z) code = 5;
    else if (z >= b.hi_z) code = 6;
    codes[gid] = code;
}

kernel void distributed_unpack_x_face(
    device const float* in_face [[buffer(0)]],
    device float* field [[buffer(1)]],
    constant HaloPackParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = p.halo * p.gy * p.gz;
    if (gid >= n) return;
    uint yz = p.gy * p.gz;
    uint hx = gid / yz;
    uint rem = gid - hx * yz;
    uint y = rem / p.gz;
    uint z = rem - y * p.gz;
    uint x = p.start + hx;
    uint dst = x * yz + y * p.gz + z;
    field[dst] = in_face[gid];
}

kernel void distributed_unpack_y_face(
    device const float* in_face [[buffer(0)]],
    device float* field [[buffer(1)]],
    constant HaloPackParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = p.gx * p.halo * p.gz;
    if (gid >= n) return;
    uint x = gid / (p.halo * p.gz);
    uint rem = gid - x * (p.halo * p.gz);
    uint hy = rem / p.gz;
    uint z = rem - hy * p.gz;
    uint y = p.start + hy;
    uint yz = p.gy * p.gz;
    uint dst = x * yz + y * p.gz + z;
    field[dst] = in_face[gid];
}

kernel void distributed_unpack_z_face(
    device const float* in_face [[buffer(0)]],
    device float* field [[buffer(1)]],
    constant HaloPackParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = p.gx * p.gy * p.halo;
    if (gid >= n) return;
    uint x = gid / (p.gy * p.halo);
    uint rem = gid - x * (p.gy * p.halo);
    uint y = rem / p.halo;
    uint hz = rem - y * p.halo;
    uint z = p.start + hz;
    uint yz = p.gy * p.gz;
    uint dst = x * yz + y * p.gz + z;
    field[dst] = in_face[gid];
}

kernel void distributed_jacobi_halo(
    device const float* phi [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device const float* halo_xm [[buffer(2)]],
    device const float* halo_xp [[buffer(3)]],
    device const float* halo_ym [[buffer(4)]],
    device const float* halo_yp [[buffer(5)]],
    device const float* halo_zm [[buffer(6)]],
    device const float* halo_zp [[buffer(7)]],
    device float* out_phi [[buffer(8)]],
    constant JacobiHaloParams& p [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = p.gx * p.gy * p.gz;
    if (gid >= n) return;

    uint yz = p.gy * p.gz;
    uint x = gid / yz;
    uint rem = gid - x * yz;
    uint y = rem / p.gz;
    uint z = rem - y * p.gz;

    uint x_halo_idx = y * p.gz + z;
    uint y_halo_idx = x * p.gz + z;
    uint z_halo_idx = x * p.gy + y;

    float xm = (x > 0) ? phi[gid - yz] : halo_xm[x_halo_idx];
    float xp = (x + 1 < p.gx) ? phi[gid + yz] : halo_xp[x_halo_idx];
    float ym = (y > 0) ? phi[gid - p.gz] : halo_ym[y_halo_idx];
    float yp = (y + 1 < p.gy) ? phi[gid + p.gz] : halo_yp[y_halo_idx];
    float zm = (z > 0) ? phi[gid - 1] : halo_zm[z_halo_idx];
    float zp = (z + 1 < p.gz) ? phi[gid + 1] : halo_zp[z_halo_idx];

    float h2 = p.dx * p.dx;
    out_phi[gid] = (xm + xp + ym + yp + zm + zp - h2 * rhs[gid]) / 6.0f;
}

kernel void distributed_advance_interior_halo(
    device const float* rho_ext [[buffer(0)]],
    device const float* mx_ext [[buffer(1)]],
    device const float* my_ext [[buffer(2)]],
    device const float* mz_ext [[buffer(3)]],
    device const float* e_ext [[buffer(4)]],
    device const float* phi_ext [[buffer(5)]],
    device float* out_rho [[buffer(6)]],
    device float* out_mx [[buffer(7)]],
    device float* out_my [[buffer(8)]],
    device float* out_mz [[buffer(9)]],
    device float* out_e [[buffer(10)]],
    constant AdvanceInteriorParams& p [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = p.gx * p.gy * p.gz;
    if (gid >= n) return;

    uint yz = p.gy * p.gz;
    uint x = gid / yz;
    uint rem = gid - x * yz;
    uint y = rem / p.gz;
    uint z = rem - y * p.gz;

    uint ey = p.gy + 2u * p.h;
    uint ez = p.gz + 2u * p.h;
    uint xh = x + p.h;
    uint yh = y + p.h;
    uint zh = z + p.h;
    uint ext = xh * ey * ez + yh * ez + zh;
    uint x_stride = ey * ez;
    uint y_stride = ez;

    uint xm = ext - x_stride;
    uint xp = ext + x_stride;
    uint ym = ext - y_stride;
    uint yp = ext + y_stride;
    uint zm = ext - 1u;
    uint zp = ext + 1u;

    float rho_c = max(rho_ext[ext], p.rho_min);
    float mx_c = mx_ext[ext];
    float my_c = my_ext[ext];
    float mz_c = mz_ext[ext];
    float e_c = e_ext[ext];

    float p_xm = (p.gamma - 1.0f) * max(e_ext[xm], 0.0f);
    float p_xp = (p.gamma - 1.0f) * max(e_ext[xp], 0.0f);
    float p_ym = (p.gamma - 1.0f) * max(e_ext[ym], 0.0f);
    float p_yp = (p.gamma - 1.0f) * max(e_ext[yp], 0.0f);
    float p_zm = (p.gamma - 1.0f) * max(e_ext[zm], 0.0f);
    float p_zp = (p.gamma - 1.0f) * max(e_ext[zp], 0.0f);

    float inv_2dx = 0.5f / p.dx;
    float grad_px = (p_xp - p_xm) * inv_2dx;
    float grad_py = (p_yp - p_ym) * inv_2dx;
    float grad_pz = (p_zp - p_zm) * inv_2dx;

    float grad_phix = (phi_ext[xp] - phi_ext[xm]) * inv_2dx;
    float grad_phiy = (phi_ext[yp] - phi_ext[ym]) * inv_2dx;
    float grad_phiz = (phi_ext[zp] - phi_ext[zm]) * inv_2dx;

    float div_m = ((mx_ext[xp] - mx_ext[xm])
                 + (my_ext[yp] - my_ext[ym])
                 + (mz_ext[zp] - mz_ext[zm])) * inv_2dx;

    float inv_dx2 = 1.0f / (p.dx * p.dx);
    float lap_e = (e_ext[xm] + e_ext[xp] + e_ext[ym] + e_ext[yp] + e_ext[zm] + e_ext[zp] - 6.0f * e_c) * inv_dx2;
    float lap_mx = (mx_ext[xm] + mx_ext[xp] + mx_ext[ym] + mx_ext[yp] + mx_ext[zm] + mx_ext[zp] - 6.0f * mx_c) * inv_dx2;
    float lap_my = (my_ext[xm] + my_ext[xp] + my_ext[ym] + my_ext[yp] + my_ext[zm] + my_ext[zp] - 6.0f * my_c) * inv_dx2;
    float lap_mz = (mz_ext[xm] + mz_ext[xp] + mz_ext[ym] + mz_ext[yp] + mz_ext[zm] + mz_ext[zp] - 6.0f * mz_c) * inv_dx2;

    out_rho[gid] = max(rho_c - p.dt * div_m, p.rho_min);
    out_mx[gid] = mx_c + p.dt * (-grad_px - rho_c * grad_phix + p.viscosity * lap_mx);
    out_my[gid] = my_c + p.dt * (-grad_py - rho_c * grad_phiy + p.viscosity * lap_my);
    out_mz[gid] = mz_c + p.dt * (-grad_pz - rho_c * grad_phiz + p.viscosity * lap_mz);
    out_e[gid] = max(e_c + p.dt * (p.thermal_diff * lap_e), 0.0f);
}
