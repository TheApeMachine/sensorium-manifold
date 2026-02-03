"""Triton kernels for Resonant Manifold (CUDA).

Implements the spectral carrier layer:
- carrier update + split (with crystallization + anchored memory)
- oscillator phase update (with top-down anchor phase bias)
- top-down energy bias from crystallized carriers
- spawn uncoupled oscillators into carriers

These kernels mirror the logic in `optimizer/metal/manifold_physics.metal`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except Exception as e:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR: Exception = e
else:
    _TRITON_IMPORT_ERROR = RuntimeError("unreachable")


ANCHORS: int = 8

STATE_VOLATILE = 0
STATE_STABLE = 1
STATE_CRYSTALLIZED = 2


@dataclass(frozen=True, slots=True)
class SpectralParams:
    # core
    dt: float
    coupling_scale: float
    carrier_reg: float
    temperature: float
    rng_seed: int
    conflict_threshold: float
    offender_weight_floor: float
    gate_width_min: float
    gate_width_max: float
    ema_alpha: float
    recenter_alpha: float
    # modes/memory
    mode: int  # 0=online,1=consolidate,2=disambiguate,3=explore
    anchor_random_eps: float
    stable_amp_threshold: float
    crystallize_amp_threshold: float
    crystallize_conflict_threshold: float
    crystallize_age: int
    crystallized_coupling_boost: float
    volatile_decay_mul: float
    stable_decay_mul: float
    crystallized_decay_mul: float
    topdown_phase_scale: float
    topdown_energy_scale: float
    topdown_random_energy_eps: float
    repulsion_scale: float


def _require_triton() -> None:
    if triton is None or tl is None:  # pragma: no cover
        raise RuntimeError(f"Triton is required for CUDA backend: {_TRITON_IMPORT_ERROR!r}")


@triton.jit
def _hash_u32(x: tl.uint32) -> tl.uint32:
    x ^= x >> 16
    x *= 0x7FEB352D
    x ^= x >> 15
    x *= 0x846CA68B
    x ^= x >> 16
    return x


@triton.jit
def _u01_from_u32(x: tl.uint32) -> tl.float32:
    # map to (0,1), avoid exact 0
    u = (x & 0x00FFFFFF).to(tl.float32) * (1.0 / 16777216.0)
    return tl.maximum(u, 1e-7)


@triton.jit
def _box_muller(u1: tl.float32, u2: tl.float32) -> tl.float32:
    r = tl.sqrt(-2.0 * tl.log(u1))
    t = 6.283185307179586 * u2
    return r * tl.cos(t)


@triton.jit
def _randn1(seed: tl.uint32, idx: tl.uint32) -> tl.float32:
    s0 = _hash_u32(seed ^ (idx * 0x9E3779B9))
    s1 = _hash_u32(s0 + 1)
    return _box_muller(_u01_from_u32(s0), _u01_from_u32(s1))


@triton.jit
def _tuning(omega_i: tl.float32, omega_k: tl.float32, gate_w: tl.float32) -> tl.float32:
    d = omega_i - omega_k
    sigma = tl.maximum(gate_w, 1e-4)
    return tl.exp(-(d * d) / (sigma * sigma))


@triton.jit
def _wrap_pi(x: tl.float32) -> tl.float32:
    # wrap to [-pi, pi]
    two_pi = 6.283185307179586
    pi = 3.141592653589793
    return x - two_pi * tl.floor((x + pi) / two_pi)


@triton.jit
def carrier_block_accum_kernel(
    osc_phase_ptr,  # fp32 [N]
    osc_omega_ptr,  # fp32 [N]
    osc_amp_ptr,  # fp32 [N]
    carrier_omega_ptr,  # fp32 [M]
    carrier_gate_ptr,  # fp32 [M]
    # accumulators (atomic adds)
    out_force_r_ptr,  # fp32 [M]
    out_force_i_ptr,  # fp32 [M]
    out_w_sum_ptr,  # fp32 [M]
    out_w_omega_sum_ptr,  # fp32 [M]
    out_w_amp_sum_ptr,  # fp32 [M]
    # sizes
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # params needed for weights
    offender_weight_floor: tl.constexpr,
    gate_width_min: tl.constexpr,
    gate_width_max: tl.constexpr,
):
    k = tl.program_id(0)
    blk = tl.program_id(1)
    omega_k = tl.load(carrier_omega_ptr + k)
    gate_w = tl.load(carrier_gate_ptr + k)
    gate_w = tl.maximum(tl.minimum(gate_w, gate_width_max), gate_width_min)
    off = blk * BLOCK_N
    idx = off + tl.arange(0, BLOCK_N)
    m = idx < N
    omega_i = tl.load(osc_omega_ptr + idx, mask=m, other=0.0)
    amp_i = tl.load(osc_amp_ptr + idx, mask=m, other=0.0)
    phi_i = tl.load(osc_phase_ptr + idx, mask=m, other=0.0)
    t = _tuning(omega_i, omega_k, gate_w)
    w = t * amp_i
    w = tl.where(w > offender_weight_floor, w, 0.0)
    zr = amp_i * tl.cos(phi_i)
    zi = amp_i * tl.sin(phi_i)
    tl.atomic_add(out_force_r_ptr + k, tl.sum(w * zr, axis=0))
    tl.atomic_add(out_force_i_ptr + k, tl.sum(w * zi, axis=0))
    tl.atomic_add(out_w_sum_ptr + k, tl.sum(w, axis=0))
    tl.atomic_add(out_w_omega_sum_ptr + k, tl.sum(w * omega_i, axis=0))
    tl.atomic_add(out_w_amp_sum_ptr + k, tl.sum(w * amp_i, axis=0))


@triton.jit
def carrier_finalize_and_split_kernel(
    osc_phase_ptr,
    osc_omega_ptr,
    osc_amp_ptr,
    carrier_real_ptr,
    carrier_imag_ptr,
    carrier_omega_ptr,
    carrier_gate_ptr,
    carrier_conflict_ptr,
    carrier_state_ptr,
    carrier_age_ptr,
    anchor_idx_ptr,
    anchor_phase_ptr,
    anchor_weight_ptr,
    num_carriers_ptr,  # int32[1] atomic
    spawned_from_ptr,
    random_phases_ptr,
    energy_stats_ptr,
    # accumulators
    force_r_ptr,
    force_i_ptr,
    w_sum_ptr,
    w_omega_sum_ptr,
    w_amp_sum_ptr,
    N: tl.constexpr,
    max_carriers: tl.constexpr,
    # params
    dt: tl.constexpr,
    carrier_reg: tl.constexpr,
    temperature: tl.constexpr,
    rng_seed: tl.constexpr,
    conflict_threshold: tl.constexpr,
    gate_width_min: tl.constexpr,
    gate_width_max: tl.constexpr,
    ema_alpha: tl.constexpr,
    recenter_alpha: tl.constexpr,
    mode: tl.constexpr,
    anchor_random_eps: tl.constexpr,
    stable_amp_threshold: tl.constexpr,
    crystallize_amp_threshold: tl.constexpr,
    crystallize_conflict_threshold: tl.constexpr,
    crystallize_age: tl.constexpr,
    volatile_decay_mul: tl.constexpr,
    stable_decay_mul: tl.constexpr,
    crystallized_decay_mul: tl.constexpr,
    repulsion_scale: tl.constexpr,
):
    k = tl.program_id(0)
    cr = tl.load(carrier_real_ptr + k)
    ci = tl.load(carrier_imag_ptr + k)
    omega_k = tl.load(carrier_omega_ptr + k)
    gate_w = tl.load(carrier_gate_ptr + k)
    gate_w = tl.maximum(tl.minimum(gate_w, gate_width_max), gate_width_min)

    state = tl.load(carrier_state_ptr + k).to(tl.int32)
    age = tl.load(carrier_age_ptr + k).to(tl.int32)

    mean_abs_e = tl.load(energy_stats_ptr + 0)
    e_scale = tl.maximum(mean_abs_e, 1e-8)
    adaptive_decay = tl.exp(-(dt) / e_scale)
    decay_mul = tl.full((), 1.0, tl.float32)
    decay_mul = tl.where(state == STATE_VOLATILE, tl.maximum(volatile_decay_mul, 0.0), decay_mul)
    decay_mul = tl.where(state == STATE_STABLE, tl.maximum(stable_decay_mul, 0.0), decay_mul)
    decay_mul = tl.where(state == STATE_CRYSTALLIZED, tl.maximum(crystallized_decay_mul, 0.0), decay_mul)
    cr = cr * adaptive_decay * decay_mul
    ci = ci * adaptive_decay * decay_mul

    force_r_raw = tl.load(force_r_ptr + k)
    force_i_raw = tl.load(force_i_ptr + k)
    w_sum = tl.load(w_sum_ptr + k)
    w_omega_sum = tl.load(w_omega_sum_ptr + k)
    w_amp_sum = tl.load(w_amp_sum_ptr + k)

    mean_omega = tl.where(w_sum > 1e-8, w_omega_sum / w_sum, omega_k)

    R = tl.sqrt(force_r_raw * force_r_raw + force_i_raw * force_i_raw)
    denom = tl.maximum(w_amp_sum, 1e-8)
    coherence = tl.maximum(tl.minimum(R / denom, 1.0), 0.0)
    inst_conflict = 1.0 - coherence

    prev_conflict = tl.load(carrier_conflict_ptr + k)
    a = tl.maximum(tl.minimum(ema_alpha, 1.0), 0.0)
    conflict = prev_conflict * (1.0 - a) + inst_conflict * a
    tl.store(carrier_conflict_ptr + k, conflict)

    if recenter_alpha != 0.0:
        rc = tl.maximum(tl.minimum(recenter_alpha, 1.0), 0.0)
        omega_k = tl.where(state == STATE_CRYSTALLIZED, omega_k, omega_k * (1.0 - rc) + mean_omega * rc)

    # metabolic shrink (skip for crystallized)
    inv_w = 1.0 / tl.maximum(w_sum, 1e-8)
    force_r = force_r_raw * inv_w
    force_i = force_i_raw * inv_w
    income = tl.sqrt(force_r * force_r + force_i * force_i)
    expense = e_scale
    not_crys = state != STATE_CRYSTALLIZED
    shrink = tl.where(not_crys & (income < expense), tl.exp(-(dt) * (expense - income) / (expense + 1e-8)), 1.0)
    cr = cr * shrink
    ci = ci * shrink

    # Langevin carrier update
    n = _randn1(rng_seed ^ 0xA5A5A5A5, k.to(tl.uint32))
    n2 = _randn1(rng_seed ^ 0xA5A5A5A5, (k.to(tl.uint32) + 1337))
    temp_factor = e_scale / (e_scale + 1.0)
    noise_scale = tl.sqrt(tl.maximum(2.0 * (temperature * temp_factor) * dt, 0.0))
    reg = tl.where(state == STATE_CRYSTALLIZED, 0.0, carrier_reg)
    cr = cr + (force_r - reg * cr) * dt + noise_scale * n
    ci = ci + (force_i - reg * ci) * dt + noise_scale * n2

    # ω repulsion (disambiguate)
    if mode == 2 and repulsion_scale > 0.0:
        curM = tl.load(num_carriers_ptr).to(tl.int32)
        repel = tl.zeros((), tl.float32)
        for k2 in range(0, max_carriers):
            active2 = (k2 < curM) & (k2 != k)
            omega2 = tl.load(carrier_omega_ptr + k2, mask=active2, other=0.0)
            gate2 = tl.load(carrier_gate_ptr + k2, mask=active2, other=0.0)
            gate2 = tl.maximum(tl.minimum(gate2, gate_width_max), gate_width_min)
            d2 = omega_k - omega2
            s = tl.maximum(gate_w + gate2, 1e-3)
            repel += tl.where(active2, d2 * tl.exp(-(d2 * d2) / (s * s)), 0.0)
        omega_k = tl.where(not_crys, omega_k + dt * repulsion_scale * repel, omega_k)

    # crystallization state machine (branchless)
    ampC = tl.sqrt(cr * cr + ci * ci)
    promote = (state == STATE_VOLATILE) & (ampC >= stable_amp_threshold)
    state = tl.where(promote, tl.full((), STATE_STABLE, tl.int32), state)
    age = tl.where(promote, 0, age)

    in_stable = state == STATE_STABLE
    ok = in_stable & (ampC >= crystallize_amp_threshold) & (conflict <= crystallize_conflict_threshold)
    age_next = tl.where(ok, age + 1, tl.where(in_stable, 0, age))
    crystallize = ok & (age_next >= crystallize_age)
    state = tl.where(crystallize, tl.full((), STATE_CRYSTALLIZED, tl.int32), state)
    age = tl.where(crystallize, crystallize_age, age_next)

    # anchor refresh (ε-greedy) using random candidate + stochastic offender
    do_anchor = (state != STATE_CRYSTALLIZED)
    h = _hash_u32((rng_seed ^ (k.to(tl.uint32) * 0xB4B82E39) ^ 0x1C3A5F7D).to(tl.uint32))
    u = _u01_from_u32(h)
    slot = (_hash_u32(h + 0x3C6EF372) % ANCHORS).to(tl.int32)
    base = k * ANCHORS + slot
    eps_anchor = tl.maximum(tl.minimum(anchor_random_eps, 1.0), 0.0)
    cand = (_hash_u32(h + 0x9E3779B9) % N).to(tl.int32)
    offender = (_hash_u32(h + 0x7F4A7C15) % N).to(tl.int32)
    chosen = tl.where(u <= eps_anchor, cand, offender)
    tl.store(anchor_idx_ptr + base, chosen, mask=do_anchor)
    psi = tl.atan2(ci, cr)
    phi_ch = tl.load(osc_phase_ptr + chosen, mask=do_anchor, other=0.0)
    d = _wrap_pi(phi_ch - psi)
    tl.store(anchor_phase_ptr + base, d, mask=do_anchor)
    omega_ch = tl.load(osc_omega_ptr + chosen, mask=do_anchor, other=0.0)
    amp_ch = tl.load(osc_amp_ptr + chosen, mask=do_anchor, other=0.0)
    w_ch = _tuning(omega_ch, omega_k, gate_w) * amp_ch
    tl.store(anchor_weight_ptr + base, w_ch, mask=do_anchor)

    # write back
    tl.store(carrier_real_ptr + k, cr)
    tl.store(carrier_imag_ptr + k, ci)
    tl.store(carrier_omega_ptr + k, omega_k)
    tl.store(carrier_gate_ptr + k, gate_w)
    tl.store(carrier_state_ptr + k, state.to(tl.int32))
    tl.store(carrier_age_ptr + k, age.to(tl.int32))

    # split: spawn when conflict is high; offender chosen stochastically.
    do_split = (state != STATE_CRYSTALLIZED) & (conflict > conflict_threshold) & (w_sum > 1e-8)
    h2 = _hash_u32((rng_seed ^ (k.to(tl.uint32) * 0xA511E9B3) ^ 0x63D83595).to(tl.uint32))
    offender_idx = (_hash_u32(h2 + 0x9E3779B9) % N).to(tl.int32)
    inc = tl.where(do_split, 1, 0).to(tl.int32)
    slot = tl.atomic_add(num_carriers_ptr, inc)
    do_write = do_split & (slot < max_carriers)
    omega_new = tl.load(osc_omega_ptr + offender_idx, mask=do_write, other=0.0)
    amp_new = tl.load(osc_amp_ptr + offender_idx, mask=do_write, other=0.0)
    phi_new = tl.load(osc_phase_ptr + offender_idx, mask=do_write, other=0.0)
    init_scale = 0.5
    nr = init_scale * amp_new * tl.cos(phi_new)
    ni = init_scale * amp_new * tl.sin(phi_new)
    tl.store(carrier_real_ptr + slot, nr, mask=do_write)
    tl.store(carrier_imag_ptr + slot, ni, mask=do_write)
    tl.store(carrier_omega_ptr + slot, omega_new, mask=do_write)
    tl.store(carrier_gate_ptr + slot, gate_w, mask=do_write)
    tl.store(carrier_conflict_ptr + slot, 0.0, mask=do_write)
    tl.store(carrier_state_ptr + slot, tl.full((), STATE_VOLATILE, tl.int32), mask=do_write)
    tl.store(carrier_age_ptr + slot, 0, mask=do_write)
    tl.store(spawned_from_ptr + slot, offender_idx, mask=do_write)
    for j in range(0, ANCHORS):
        b = slot * ANCHORS + j
        tl.store(anchor_idx_ptr + b, tl.full((), -1, tl.int32), mask=do_write)
        tl.store(anchor_phase_ptr + b, 0.0, mask=do_write)
        tl.store(anchor_weight_ptr + b, 0.0, mask=do_write)
    b0 = slot * ANCHORS
    tl.store(anchor_idx_ptr + b0, offender_idx, mask=do_write)
    psi0 = tl.atan2(ni, nr)
    d0 = _wrap_pi(phi_new - psi0)
    tl.store(anchor_phase_ptr + b0, d0, mask=do_write)
    tl.store(anchor_weight_ptr + b0, amp_new, mask=do_write)
    rp = tl.load(random_phases_ptr + slot, mask=do_write, other=0.0)
    r = rp * 6.283185307179586
    rot_r = tl.cos(r)
    rot_i = tl.sin(r)
    rr = tl.load(carrier_real_ptr + slot, mask=do_write, other=0.0)
    ri = tl.load(carrier_imag_ptr + slot, mask=do_write, other=0.0)
    tl.store(carrier_real_ptr + slot, rr * rot_r - ri * rot_i, mask=do_write)
    tl.store(carrier_imag_ptr + slot, rr * rot_i + ri * rot_r, mask=do_write)
    tl.store(carrier_conflict_ptr + k, 0.0, mask=do_split)


@triton.jit
def spectral_update_oscillator_phases_kernel(
    osc_phase_ptr,
    osc_omega_ptr,
    osc_amp_ptr,
    carrier_real_ptr,
    carrier_imag_ptr,
    carrier_omega_ptr,
    carrier_gate_ptr,
    carrier_state_ptr,
    anchor_idx_ptr,
    anchor_phase_ptr,
    anchor_weight_ptr,
    energy_stats_ptr,
    num_carriers_ptr,  # int32 [1]
    N: tl.constexpr,
    max_carriers: tl.constexpr,
    # params
    dt: tl.constexpr,
    coupling_scale: tl.constexpr,
    temperature: tl.constexpr,
    rng_seed: tl.constexpr,
    gate_width_min: tl.constexpr,
    gate_width_max: tl.constexpr,
    crystallized_coupling_boost: tl.constexpr,
    topdown_phase_scale: tl.constexpr,
):
    i = tl.program_id(0)
    phi = tl.load(osc_phase_ptr + i)
    omega_i = tl.load(osc_omega_ptr + i)
    amp_i = tl.load(osc_amp_ptr + i)

    num_car = tl.load(num_carriers_ptr).to(tl.int32)
    torque = tl.zeros((), tl.float32)
    for k in range(0, max_carriers):
        active = k < num_car
        omega_k = tl.load(carrier_omega_ptr + k, mask=active, other=0.0)
        gate_w = tl.load(carrier_gate_ptr + k, mask=active, other=1.0)
        gate_w = tl.maximum(tl.minimum(gate_w, gate_width_max), gate_width_min)
        t = _tuning(omega_i, omega_k, gate_w)
        cr = tl.load(carrier_real_ptr + k, mask=active, other=0.0)
        ci = tl.load(carrier_imag_ptr + k, mask=active, other=0.0)
        psi = tl.atan2(ci, cr)
        R = tl.sqrt(cr * cr + ci * ci)
        st = tl.load(carrier_state_ptr + k, mask=active, other=0).to(tl.int32)
        boost = 1.0 + tl.where(st == STATE_CRYSTALLIZED, tl.maximum(crystallized_coupling_boost, 0.0), 0.0)
        torque += boost * t * (amp_i * R) * tl.sin(psi - phi)

        # top-down phase pull if anchored in crystallized carrier
        if topdown_phase_scale != 0.0:
            base = k * ANCHORS
            is_crys = st == STATE_CRYSTALLIZED
            for j in range(0, ANCHORS):
                idx = tl.load(anchor_idx_ptr + base + j, mask=active, other=-1).to(tl.int32)
                match = is_crys & (idx == i)
                off = tl.load(anchor_phase_ptr + base + j, mask=match, other=0.0)
                w = tl.load(anchor_weight_ptr + base + j, mask=match, other=0.0)
                target = psi + off
                d = _wrap_pi(target - phi)
                torque += topdown_phase_scale * w * tl.sin(d)

    mean_abs_e = tl.load(energy_stats_ptr + 0)
    e_scale = tl.maximum(mean_abs_e, 1e-8)
    temp_factor = e_scale / (e_scale + 1.0)
    noise_scale = tl.sqrt(tl.maximum(2.0 * (temperature * temp_factor) * dt, 0.0))
    n = _randn1(rng_seed ^ 0xC3C3C3C3, i.to(tl.uint32))
    dphi = omega_i + coupling_scale * torque
    phi = phi + dphi * dt + noise_scale * n
    two_pi = 6.283185307179586
    phi = phi - two_pi * tl.floor(phi / two_pi)
    tl.store(osc_phase_ptr + i, phi)


@triton.jit
def spectral_topdown_bias_energies_kernel(
    osc_energy_ptr,  # fp32 [N]
    osc_amp_ptr,  # fp32 [N]
    carrier_state_ptr,  # int32 [M]
    anchor_idx_ptr,  # int32 [M*ANCHORS]
    anchor_weight_ptr,  # fp32 [M*ANCHORS]
    num_carriers_ptr,  # int32 [1]
    N: tl.constexpr,
    max_carriers: tl.constexpr,
    dt: tl.constexpr,
    rng_seed: tl.constexpr,
    topdown_energy_scale: tl.constexpr,
    topdown_random_energy_eps: tl.constexpr,
):
    k = tl.program_id(0)
    num_car = tl.load(num_carriers_ptr).to(tl.int32)
    active = k < num_car
    st = tl.load(carrier_state_ptr + k, mask=active, other=0).to(tl.int32)
    is_crys = active & (st == STATE_CRYSTALLIZED) & (topdown_energy_scale > 0.0)
    base = k * ANCHORS
    wsum = tl.zeros((), tl.float32)
    act = tl.zeros((), tl.float32)
    for j in range(0, ANCHORS):
        idx = tl.load(anchor_idx_ptr + base + j, mask=is_crys, other=-1).to(tl.int32)
        ok = is_crys & (idx >= 0) & (idx < N)
        w = tl.load(anchor_weight_ptr + base + j, mask=ok, other=0.0)
        wsum += w
        act += w * tl.load(osc_amp_ptr + idx, mask=ok, other=0.0)
    ok_w = is_crys & (wsum > 1e-8)
    act = act / wsum
    for j in range(0, ANCHORS):
        idx = tl.load(anchor_idx_ptr + base + j, mask=ok_w, other=-1).to(tl.int32)
        ok = ok_w & (idx >= 0) & (idx < N)
        w = tl.load(anchor_weight_ptr + base + j, mask=ok, other=0.0) / wsum
        a = tl.load(osc_amp_ptr + idx, mask=ok, other=0.0)
        need = 1.0 / (1.0 + a)
        dE = dt * topdown_energy_scale * act * w * need
        tl.atomic_add(osc_energy_ptr + idx, dE, mask=ok)

    if topdown_random_energy_eps > 0.0:
        h = _hash_u32((rng_seed ^ (k.to(tl.uint32) * 0x27D4EB2D) ^ 0x85EBCA6B).to(tl.uint32))
        u = _u01_from_u32(h)
        lucky = ok_w & (u <= topdown_random_energy_eps)
        idx = (_hash_u32(h + 0x165667B1) % N).to(tl.int32)
        dE = dt * (0.25 * topdown_energy_scale) * act
        tl.atomic_add(osc_energy_ptr + idx, dE, mask=lucky)


@triton.jit
def spectral_spawn_uncoupled_kernel(
    osc_phase_ptr,
    osc_omega_ptr,
    osc_amp_ptr,
    carrier_real_ptr,
    carrier_imag_ptr,
    carrier_omega_ptr,
    carrier_gate_ptr,
    carrier_conflict_ptr,
    carrier_state_ptr,
    carrier_age_ptr,
    anchor_idx_ptr,
    anchor_phase_ptr,
    anchor_weight_ptr,
    num_carriers_ptr,  # int32[1] atomic
    N: tl.constexpr,
    max_carriers: tl.constexpr,
    coupling_threshold: tl.constexpr,
    gate_width_init: tl.constexpr,
    gate_width_min: tl.constexpr,
    gate_width_max: tl.constexpr,
):
    i = tl.program_id(0)
    omega_i = tl.load(osc_omega_ptr + i)
    amp_i = tl.load(osc_amp_ptr + i)
    phi_i = tl.load(osc_phase_ptr + i)

    num_car = tl.load(num_carriers_ptr).to(tl.int32)
    total = tl.zeros((), tl.float32)
    for k in range(0, max_carriers):
        active = k < num_car
        omega_k = tl.load(carrier_omega_ptr + k, mask=active, other=0.0)
        gate_w = tl.load(carrier_gate_ptr + k, mask=active, other=1.0)
        gate_w = tl.maximum(tl.minimum(gate_w, gate_width_max), gate_width_min)
        total += _tuning(omega_i, omega_k, gate_w)
    do_spawn = total < coupling_threshold
    inc = tl.where(do_spawn, 1, 0).to(tl.int32)
    slot = tl.atomic_add(num_carriers_ptr, inc)
    do_write = do_spawn & (slot < max_carriers)
    tl.store(carrier_real_ptr + slot, amp_i * tl.cos(phi_i), mask=do_write)
    tl.store(carrier_imag_ptr + slot, amp_i * tl.sin(phi_i), mask=do_write)
    tl.store(carrier_omega_ptr + slot, omega_i, mask=do_write)
    tl.store(carrier_gate_ptr + slot, gate_width_init, mask=do_write)
    tl.store(carrier_conflict_ptr + slot, 0.0, mask=do_write)
    tl.store(carrier_state_ptr + slot, tl.full((), STATE_VOLATILE, tl.int32), mask=do_write)
    tl.store(carrier_age_ptr + slot, 0, mask=do_write)
    for j in range(0, ANCHORS):
        b = slot * ANCHORS + j
        tl.store(anchor_idx_ptr + b, tl.full((), -1, tl.int32), mask=do_write)
        tl.store(anchor_phase_ptr + b, 0.0, mask=do_write)
        tl.store(anchor_weight_ptr + b, 0.0, mask=do_write)
    b0 = slot * ANCHORS
    tl.store(anchor_idx_ptr + b0, i, mask=do_write)
    tl.store(anchor_phase_ptr + b0, 0.0, mask=do_write)
    tl.store(anchor_weight_ptr + b0, amp_i, mask=do_write)


def carrier_update_and_split(
    *,
    osc_phase: torch.Tensor,
    osc_omega: torch.Tensor,
    osc_amp: torch.Tensor,
    carrier_real: torch.Tensor,
    carrier_imag: torch.Tensor,
    carrier_omega: torch.Tensor,
    carrier_gate_width: torch.Tensor,
    carrier_conflict: torch.Tensor,
    carrier_state: torch.Tensor,
    carrier_age: torch.Tensor,
    anchor_idx: torch.Tensor,
    anchor_phase: torch.Tensor,
    anchor_weight: torch.Tensor,
    num_carriers: torch.Tensor,
    spawned_from: torch.Tensor,
    random_phases: torch.Tensor,
    energy_stats: torch.Tensor,
    current_carriers: int,
    max_carriers: int,
    params: SpectralParams,
) -> None:
    _require_triton()
    assert osc_phase.is_cuda
    N = int(osc_phase.numel())
    if N == 0 or current_carriers == 0:
        return
    BLOCK_N = 256
    blocks = triton.cdiv(N, BLOCK_N)

    # Accumulators (per active carrier)
    force_r = torch.zeros((current_carriers,), device=osc_phase.device, dtype=torch.float32)
    force_i = torch.zeros((current_carriers,), device=osc_phase.device, dtype=torch.float32)
    w_sum = torch.zeros((current_carriers,), device=osc_phase.device, dtype=torch.float32)
    w_omega_sum = torch.zeros((current_carriers,), device=osc_phase.device, dtype=torch.float32)
    w_amp_sum = torch.zeros((current_carriers,), device=osc_phase.device, dtype=torch.float32)

    # Pass 1: accumulate block partials into per-carrier sums
    carrier_block_accum_kernel[(current_carriers, blocks)](
        osc_phase,
        osc_omega,
        osc_amp,
        carrier_omega,
        carrier_gate_width,
        force_r,
        force_i,
        w_sum,
        w_omega_sum,
        w_amp_sum,
        N=N,
        BLOCK_N=BLOCK_N,
        offender_weight_floor=float(params.offender_weight_floor),
        gate_width_min=float(params.gate_width_min),
        gate_width_max=float(params.gate_width_max),
        num_warps=1,
    )

    # Pass 2: finalize updates + split + crystallization + anchor refresh
    carrier_finalize_and_split_kernel[(current_carriers,)](
        osc_phase,
        osc_omega,
        osc_amp,
        carrier_real,
        carrier_imag,
        carrier_omega,
        carrier_gate_width,
        carrier_conflict,
        carrier_state,
        carrier_age,
        anchor_idx,
        anchor_phase,
        anchor_weight,
        num_carriers,
        spawned_from,
        random_phases,
        energy_stats,
        force_r,
        force_i,
        w_sum,
        w_omega_sum,
        w_amp_sum,
        N=N,
        max_carriers=max_carriers,
        dt=float(params.dt),
        carrier_reg=float(params.carrier_reg),
        temperature=float(params.temperature),
        rng_seed=int(params.rng_seed),
        conflict_threshold=float(params.conflict_threshold),
        gate_width_min=float(params.gate_width_min),
        gate_width_max=float(params.gate_width_max),
        ema_alpha=float(params.ema_alpha),
        recenter_alpha=float(params.recenter_alpha),
        mode=int(params.mode),
        anchor_random_eps=float(params.anchor_random_eps),
        stable_amp_threshold=float(params.stable_amp_threshold),
        crystallize_amp_threshold=float(params.crystallize_amp_threshold),
        crystallize_conflict_threshold=float(params.crystallize_conflict_threshold),
        crystallize_age=int(params.crystallize_age),
        volatile_decay_mul=float(params.volatile_decay_mul),
        stable_decay_mul=float(params.stable_decay_mul),
        crystallized_decay_mul=float(params.crystallized_decay_mul),
        repulsion_scale=float(params.repulsion_scale),
        num_warps=1,
    )


def topdown_bias_energies(
    *,
    osc_energy: torch.Tensor,
    osc_amp: torch.Tensor,
    carrier_state: torch.Tensor,
    anchor_idx: torch.Tensor,
    anchor_weight: torch.Tensor,
    num_carriers: torch.Tensor,
    num_carriers_i: int,
    max_carriers: int,
    dt: float,
    rng_seed: int,
    topdown_energy_scale: float,
    topdown_random_energy_eps: float,
) -> None:
    _require_triton()
    N = int(osc_energy.numel())
    if N == 0 or num_carriers_i == 0:
        return
    grid = (num_carriers_i,)
    spectral_topdown_bias_energies_kernel[grid](
        osc_energy,
        osc_amp,
        carrier_state,
        anchor_idx,
        anchor_weight,
        num_carriers,
        N=N,
        max_carriers=max_carriers,
        dt=float(dt),
        rng_seed=int(rng_seed),
        topdown_energy_scale=float(topdown_energy_scale),
        topdown_random_energy_eps=float(topdown_random_energy_eps),
        num_warps=1,
    )


def update_oscillator_phases(
    *,
    osc_phase: torch.Tensor,
    osc_omega: torch.Tensor,
    osc_amp: torch.Tensor,
    carrier_real: torch.Tensor,
    carrier_imag: torch.Tensor,
    carrier_omega: torch.Tensor,
    carrier_gate_width: torch.Tensor,
    carrier_state: torch.Tensor,
    anchor_idx: torch.Tensor,
    anchor_phase: torch.Tensor,
    anchor_weight: torch.Tensor,
    energy_stats: torch.Tensor,
    num_carriers: torch.Tensor,
    N: int,
    max_carriers: int,
    dt: float,
    coupling_scale: float,
    temperature: float,
    rng_seed: int,
    gate_width_min: float,
    gate_width_max: float,
    crystallized_coupling_boost: float,
    topdown_phase_scale: float,
) -> None:
    _require_triton()
    if N == 0:
        return
    grid = (triton.cdiv(N, 1),)
    spectral_update_oscillator_phases_kernel[grid](
        osc_phase,
        osc_omega,
        osc_amp,
        carrier_real,
        carrier_imag,
        carrier_omega,
        carrier_gate_width,
        carrier_state,
        anchor_idx,
        anchor_phase,
        anchor_weight,
        energy_stats,
        num_carriers,
        N=N,
        max_carriers=max_carriers,
        dt=float(dt),
        coupling_scale=float(coupling_scale),
        temperature=float(temperature),
        rng_seed=int(rng_seed),
        gate_width_min=float(gate_width_min),
        gate_width_max=float(gate_width_max),
        crystallized_coupling_boost=float(crystallized_coupling_boost),
        topdown_phase_scale=float(topdown_phase_scale),
        num_warps=1,
    )


def spawn_uncoupled(
    *,
    osc_phase: torch.Tensor,
    osc_omega: torch.Tensor,
    osc_amp: torch.Tensor,
    carrier_real: torch.Tensor,
    carrier_imag: torch.Tensor,
    carrier_omega: torch.Tensor,
    carrier_gate_width: torch.Tensor,
    carrier_conflict: torch.Tensor,
    carrier_state: torch.Tensor,
    carrier_age: torch.Tensor,
    anchor_idx: torch.Tensor,
    anchor_phase: torch.Tensor,
    anchor_weight: torch.Tensor,
    num_carriers: torch.Tensor,
    num_carriers_i: int,
    max_carriers: int,
    coupling_threshold: float,
    gate_width_init: float,
    gate_width_min: float,
    gate_width_max: float,
) -> None:
    _require_triton()
    N = int(osc_phase.numel())
    if N == 0:
        return
    grid = (triton.cdiv(N, 1),)
    spectral_spawn_uncoupled_kernel[grid](
        osc_phase,
        osc_omega,
        osc_amp,
        carrier_real,
        carrier_imag,
        carrier_omega,
        carrier_gate_width,
        carrier_conflict,
        carrier_state,
        carrier_age,
        anchor_idx,
        anchor_phase,
        anchor_weight,
        num_carriers,
        N=N,
        max_carriers=max_carriers,
        coupling_threshold=float(coupling_threshold),
        gate_width_init=float(gate_width_init),
        gate_width_min=float(gate_width_min),
        gate_width_max=float(gate_width_max),
        num_warps=1,
    )

