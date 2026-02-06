"""Triton kernels for Resonant Manifold (CUDA).

Implements the spectral mode layer:
- mode update + split (with condensation + anchored memory)
- oscillator phase update (with top-down anchor phase bias)
- top-down energy bias from condensed modes
- spawn uncoupled oscillators into modes

These kernels mirror the logic in `optimizer/metal/manifold_physics.metal`.

═══════════════════════════════════════════════════════════════════════════════
PHYSICS-BASED DESIGN
═══════════════════════════════════════════════════════════════════════════════

All parameters derive from:
  - T: temperature from geometric domain
  - dt: simulation timestep
  - Physical constants: ℏ, k_B

No arbitrary hyperparameters. The quantum thermodynamic formulas are in
`optimizer/physics_units.py`. This file implements the kernels that use
the derived quantities.

Mode states (physics-based):
  - MODE_EXCITED (0):   Amplitude > thermal noise floor, phase incoherent
  - MODE_COHERENT (1):  Amplitude > noise floor, phase stable (locked)
  - MODE_CONDENSED (2): Macroscopic occupation, frozen (BEC-like)
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except Exception as e:  # pragma: no cover
    # Provide a dummy `triton.jit` decorator so this module can be imported
    # in non-CUDA / non-Triton environments. Runtime entrypoints still call
    # `_require_triton()` and will raise when Triton is unavailable.
    class _DummyTriton:
        @staticmethod
        def jit(fn=None, **_kwargs):  # type: ignore[no-untyped-def]
            if fn is None:
                return lambda f: f
            return fn

    triton = _DummyTriton()  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR: Exception = e
else:
    _TRITON_IMPORT_ERROR = RuntimeError("unreachable")


ANCHORS: int = 8

# Mode states (physics-based definitions)
# [CHOICE] Three states: excited, coherent, condensed
# [REASON] Maps to quantum statistical mechanics:
#   - EXCITED: above thermal noise but phase-incoherent
#   - COHERENT: phase-locked (stable oscillation)
#   - CONDENSED: macroscopic occupation (Bose-Einstein condensate)
MODE_EXCITED = 0      # Was: STATE_VOLATILE
MODE_COHERENT = 1     # Was: STATE_STABLE
MODE_CONDENSED = 2    # Was: STATE_CRYSTALLIZED

# Legacy aliases for backward compatibility during transition
STATE_VOLATILE = MODE_EXCITED
STATE_STABLE = MODE_COHERENT
STATE_CRYSTALLIZED = MODE_CONDENSED


@dataclass(frozen=True, slots=True)
class ModeParams:
    """Physics-derived parameters for mode dynamics.

    All values computed from temperature T, timestep dt, and physical constants.
    See `optimizer/physics_units.py` for the derivations.

    [DESIGN] No arbitrary hyperparameters. Each field has a physical derivation.
    """

    # ─── Core simulation parameters ───
    dt: float                   # Simulation timestep [sim time units]
    temperature: float          # Temperature from geometric domain [sim K]
    rng_seed: int              # Random seed for stochastic terms

    # ─── Physics-derived scales ───
    # [DERIVED FROM] ModePhysics.from_temperature()

    noise_floor_scale: float    # √(k_B T / m) — thermal amplitude scale
                                # [FORMULA] A_thermal ∝ noise_floor_scale / √ω
                                # [REASON] Mode visible only if A > A_thermal(ω)

    bandwidth_min: float        # 1/(2·dt) — uncertainty-limited bandwidth
                                # [FORMULA] Δω·Δt ≥ 1/2 (Heisenberg)
                                # [REASON] Cannot resolve finer frequency differences

    bandwidth_max: float        # k_B T / ℏ — thermal bandwidth
                                # [FORMULA] ω_thermal = k_B T / ℏ
                                # [REASON] Modes spread up to thermal frequency

    decay_rate: float           # Lindblad dissipation rate [1/time]
                                # [FORMULA] Γ = γ · k_B T / ℏ
                                # [REASON] Thermal bath causes amplitude decay

    coherence_steps: int        # τ_coherence / dt — steps to phase-lock
                                # [FORMULA] τ_coh = ℏ / k_B T
                                # [REASON] Time for stable phase relationship

    # ─── Coupling physics ───
    coupling_scale: float       # Interaction strength (dimensionless)
                                # [CHOICE] V_ij / k_B T — ratio of coupling to thermal energy
                                # [REASON] Determines phase-locking strength
                                # [NOTES] Should be O(0.1-1) for meaningful coupling

    # ─── Mode operation (not a hyperparameter — controls algorithm behavior) ───
    mode: int                   # 0=online, 1=consolidate, 2=disambiguate, 3=explore

    @classmethod
    def from_physics(
        cls,
        T: float,
        dt: float,
        hbar: float,
        k_B: float,
        rng_seed: int,
        *,
        bath_coupling: float = 0.1,
        effective_mass: float = 1.0,
        mode: int = 0,
    ) -> "ModeParams":
        """Compute mode parameters from physical quantities.

        Args:
            T: Temperature from geometric domain
            dt: Simulation timestep
            hbar: Reduced Planck constant (simulation units)
            k_B: Boltzmann constant (simulation units)
            rng_seed: Random seed
            bath_coupling: Coupling to thermal bath (dimensionless, default 0.1)
                [PHYSICAL MEANING] How strongly modes thermalize with environment.
                This IS a physical parameter that could be measured.
            effective_mass: Mass scale (default 1.0 in natural units)
            mode: Operation mode (0=online, 1=consolidate, 2=disambiguate, 3=explore)

        Returns:
            ModeParams with all physics-derived values.
        """
        # Protect against edge cases
        T = max(T, 1e-10)  # Avoid division by zero at T=0

        # Thermal scales
        omega_thermal = (k_B * T) / hbar
        tau_coherence = hbar / (k_B * T)

        # Amplitude scale: √(k_B T / m)
        noise_scale = math.sqrt((k_B * T) / effective_mass) if effective_mass > 0 else 1e-10

        # Bandwidth bounds (uncertainty principle)
        bw_min = 0.5 / dt
        bw_max = omega_thermal

        # Lindblad decay rate
        gamma = bath_coupling * omega_thermal

        # Coherence time in steps
        coh_steps = max(1, int(tau_coherence / dt))

        # Coupling scale: interaction / thermal energy
        # For weak coupling regime, use bath_coupling as proxy
        coupling = bath_coupling

        return cls(
            dt=dt,
            temperature=T,
            rng_seed=rng_seed,
            noise_floor_scale=noise_scale,
            bandwidth_min=bw_min,
            bandwidth_max=bw_max,
            decay_rate=gamma,
            coherence_steps=coh_steps,
            coupling_scale=coupling,
            mode=mode,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════
# SpectralParams is kept for backward compatibility during transition.
# New code should use ModeParams.from_physics().
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SpectralParams:
    """DEPRECATED: Use ModeParams instead.

    Kept for backward compatibility. All these arbitrary parameters
    should be replaced with physics-derived values from ModeParams.
    """
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


# New, clearer name for the same legacy parameter bundle.
# We keep the original name to avoid breaking external imports.
CoherenceParams = SpectralParams

# Preferred naming: this parameter bundle governs ω-space wavefluid dynamics.
HydrodynamicParams = CoherenceParams


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
    # [CHOICE] Lorentzian resonance from finite coherence time / linewidth.
    # [FORMULA] r(Δω) = γ^2 / (Δω^2 + γ^2)
    # [REASON] Physical lineshape for a damped oscillator / finite τ_coh.
    d = omega_i - omega_k
    gamma = tl.maximum(gate_w, 1e-8)
    g2 = gamma * gamma
    return g2 / (d * d + g2)


@triton.jit
def _min_image_1d(d: tl.float32, L: tl.float32) -> tl.float32:
    # Minimum image convention for periodic domain of size L.
    # Equivalent to: d - L * round(d/L)
    invL = 1.0 / tl.maximum(L, 1e-12)
    q = d * invL
    r = tl.floor(q + 0.5)
    return d - L * r


@triton.jit
def _spatial_overlap_from_anchors(
    pos_x: tl.float32,
    pos_y: tl.float32,
    pos_z: tl.float32,
    particle_pos_ptr,      # fp32 [N,3] contiguous
    anchor_idx_ptr,        # int32 [maxM*ANCHORS]
    anchor_weight_ptr,     # fp32 [maxM*ANCHORS]
    carrier_k: tl.int32,
    domain_x: tl.float32,
    domain_y: tl.float32,
    domain_z: tl.float32,
    spatial_sigma: tl.float32,
    N: tl.constexpr,
) -> tl.float32:
    # [CHOICE] Real-space overlap integral proxy (Gaussian wavepackets).
    # [FORMULA] O = Σ_a w_a exp(-|Δx|^2/(4σ_x^2)) / Σ_a w_a
    sigma = spatial_sigma
    if not (sigma > 0.0):
        return tl.zeros_like(pos_x)
    inv_4s2 = 1.0 / (4.0 * sigma * sigma)
    base = carrier_k * ANCHORS
    sum_w = tl.zeros((), tl.float32)
    sum_ov = tl.zeros_like(pos_x, tl.float32)
    for j in range(0, ANCHORS):
        idxa = tl.load(anchor_idx_ptr + base + j).to(tl.int32)
        ok = (idxa >= 0) & (idxa < N)
        w = tl.load(anchor_weight_ptr + base + j, mask=ok, other=0.0)
        ok = ok & (w > 0.0)
        ax = tl.load(particle_pos_ptr + idxa * 3 + 0, mask=ok, other=0.0)
        ay = tl.load(particle_pos_ptr + idxa * 3 + 1, mask=ok, other=0.0)
        az = tl.load(particle_pos_ptr + idxa * 3 + 2, mask=ok, other=0.0)
        dx = _min_image_1d(pos_x - ax, domain_x)
        dy = _min_image_1d(pos_y - ay, domain_y)
        dz = _min_image_1d(pos_z - az, domain_z)
        r2 = dx * dx + dy * dy + dz * dz
        ov = tl.exp(-r2 * inv_4s2)
        sum_w += tl.where(ok, w, 0.0)
        sum_ov += tl.where(ok, w * ov, 0.0)
    denom = tl.maximum(sum_w, 1e-12)
    return tl.where(sum_w > 0.0, sum_ov / denom, 0.0)


@triton.jit
def _wrap_pi(x: tl.float32) -> tl.float32:
    # wrap to [-pi, pi]
    two_pi = 6.283185307179586
    pi = 3.141592653589793
    return x - two_pi * tl.floor((x + pi) / two_pi)


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS-BASED HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit
def _thermal_noise_floor(omega: tl.float32, noise_floor_scale: tl.float32) -> tl.float32:
    """Compute thermal noise floor for a mode at frequency omega.

    [CHOICE] Classical limit approximation: A_thermal ≈ √(k_B T / m) / √ω
    [FORMULA] A_thermal(ω) = noise_floor_scale / √ω where noise_floor_scale = √(k_B T / m)
    [REASON] Valid when ℏω << k_B T (most modes). Conservative — high-freq modes
             have even smaller thermal amplitude due to zero-point energy.
    [NOTES] Returns noise_floor_scale if ω ≤ 0 (prevents division by zero).
    """
    omega_safe = tl.maximum(omega, 1e-8)
    return noise_floor_scale / tl.sqrt(omega_safe)


@triton.jit
def _visibility_ratio(amplitude: tl.float32, omega: tl.float32, noise_floor_scale: tl.float32) -> tl.float32:
    """Compute visibility ratio: amplitude / thermal_noise_floor.

    [CHOICE] Signal-to-noise ratio for mode detection
    [FORMULA] visibility = A / A_thermal(ω)
    [REASON] A mode is:
             - THERMAL (noise) if visibility ≤ 1.0
             - EXCITED if visibility > 1.0
             - STRONGLY EXCITED if visibility >> 1.0
    [NOTES] This replaces arbitrary amplitude thresholds with physics.
    """
    noise = _thermal_noise_floor(omega, noise_floor_scale)
    return amplitude / tl.maximum(noise, 1e-10)


@triton.jit
def carrier_block_accum_kernel(
    osc_phase_ptr,  # fp32 [N]
    osc_omega_ptr,  # fp32 [N]
    osc_amp_ptr,  # fp32 [N]
    particle_pos_ptr,  # fp32 [N,3]
    carrier_omega_ptr,  # fp32 [M]
    carrier_gate_ptr,  # fp32 [M]
    anchor_idx_ptr,  # int32 [M*ANCHORS]
    anchor_weight_ptr,  # fp32 [M*ANCHORS]
    # accumulators (atomic adds)
    out_force_r_ptr,  # fp32 [M]
    out_force_i_ptr,  # fp32 [M]
    out_w_sum_ptr,  # fp32 [M]
    out_w_omega_sum_ptr,  # fp32 [M]
    out_w_omega2_sum_ptr,  # fp32 [M]
    out_w_amp_sum_ptr,  # fp32 [M]
    # sizes
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # params needed for weights
    offender_weight_floor: tl.constexpr,
    gate_width_min: tl.constexpr,
    gate_width_max: tl.constexpr,
    domain_x: tl.float32,
    domain_y: tl.float32,
    domain_z: tl.float32,
    spatial_sigma: tl.float32,
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
    pos_x = tl.load(particle_pos_ptr + idx * 3 + 0, mask=m, other=0.0)
    pos_y = tl.load(particle_pos_ptr + idx * 3 + 1, mask=m, other=0.0)
    pos_z = tl.load(particle_pos_ptr + idx * 3 + 2, mask=m, other=0.0)
    r = _tuning(omega_i, omega_k, gate_w)
    s = _spatial_overlap_from_anchors(
        pos_x,
        pos_y,
        pos_z,
        particle_pos_ptr,
        anchor_idx_ptr,
        anchor_weight_ptr,
        k.to(tl.int32),
        domain_x,
        domain_y,
        domain_z,
        spatial_sigma,
        N=N,
    )
    w = (r * s) * amp_i
    w = tl.where(w > offender_weight_floor, w, 0.0)
    zr = amp_i * tl.cos(phi_i)
    zi = amp_i * tl.sin(phi_i)
    tl.atomic_add(out_force_r_ptr + k, tl.sum(w * zr, axis=0))
    tl.atomic_add(out_force_i_ptr + k, tl.sum(w * zi, axis=0))
    tl.atomic_add(out_w_sum_ptr + k, tl.sum(w, axis=0))
    tl.atomic_add(out_w_omega_sum_ptr + k, tl.sum(w * omega_i, axis=0))
    tl.atomic_add(out_w_omega2_sum_ptr + k, tl.sum(w * omega_i * omega_i, axis=0))
    tl.atomic_add(out_w_amp_sum_ptr + k, tl.sum(w * amp_i, axis=0))


@triton.jit
def carrier_finalize_and_split_kernel(
    osc_phase_ptr,
    osc_omega_ptr,
    osc_amp_ptr,
    particle_pos_ptr,  # fp32 [N,3]
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
    w_omega2_sum_ptr,
    w_amp_sum_ptr,
    domain_x: tl.float32,
    domain_y: tl.float32,
    domain_z: tl.float32,
    spatial_sigma: tl.float32,
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
    w_omega2_sum = tl.load(w_omega2_sum_ptr + k)
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

    # [CHOICE] adaptive gate width (frequency spread of current supporters)
    # [FORMULA] σ_k^2 = E_w[ω^2] - (E_w[ω])^2
    # [REASON] restore “open/sharpen” behavior as an emergent property (not conflict-driven)
    if recenter_alpha != 0.0:
        rc = tl.maximum(tl.minimum(recenter_alpha, 1.0), 0.0)
        Ew = tl.where(w_sum > 1e-8, w_omega_sum / w_sum, omega_k)
        Ew2 = tl.where(w_sum > 1e-8, w_omega2_sum / w_sum, omega_k * omega_k)
        var = tl.maximum(Ew2 - Ew * Ew, 0.0)
        sigma_hat = tl.sqrt(var)
        gate_w = tl.where(state == STATE_CRYSTALLIZED, gate_w, gate_w * (1.0 - rc) + sigma_hat * rc)
    gate_w = tl.maximum(tl.minimum(gate_w, gate_width_max), gate_width_min)

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
    psi = tl.atan2(ci, cr)
    phi_ch = tl.load(osc_phase_ptr + chosen, mask=do_anchor, other=0.0)
    d = _wrap_pi(phi_ch - psi)
    tl.store(anchor_phase_ptr + base, d, mask=do_anchor)
    omega_ch = tl.load(osc_omega_ptr + chosen, mask=do_anchor, other=0.0)
    amp_ch = tl.load(osc_amp_ptr + chosen, mask=do_anchor, other=0.0)
    pos_x = tl.load(particle_pos_ptr + chosen * 3 + 0, mask=do_anchor, other=0.0)
    pos_y = tl.load(particle_pos_ptr + chosen * 3 + 1, mask=do_anchor, other=0.0)
    pos_z = tl.load(particle_pos_ptr + chosen * 3 + 2, mask=do_anchor, other=0.0)
    r_ch = _tuning(omega_ch, omega_k, gate_w)
    s_ch = _spatial_overlap_from_anchors(
        pos_x,
        pos_y,
        pos_z,
        particle_pos_ptr,
        anchor_idx_ptr,
        anchor_weight_ptr,
        k.to(tl.int32),
        domain_x,
        domain_y,
        domain_z,
        spatial_sigma,
        N=N,
    )
    w_ch = (r_ch * s_ch) * amp_ch
    tl.store(anchor_idx_ptr + base, chosen, mask=do_anchor)
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
    particle_pos_ptr,  # fp32 [N,3]
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
    domain_x: tl.float32,
    domain_y: tl.float32,
    domain_z: tl.float32,
    spatial_sigma: tl.float32,
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
    pos_x = tl.load(particle_pos_ptr + i * 3 + 0)
    pos_y = tl.load(particle_pos_ptr + i * 3 + 1)
    pos_z = tl.load(particle_pos_ptr + i * 3 + 2)

    num_car = tl.load(num_carriers_ptr).to(tl.int32)
    torque = tl.zeros((), tl.float32)
    for k in range(0, max_carriers):
        active = k < num_car
        omega_k = tl.load(carrier_omega_ptr + k, mask=active, other=0.0)
        gate_w = tl.load(carrier_gate_ptr + k, mask=active, other=1.0)
        gate_w = tl.maximum(tl.minimum(gate_w, gate_width_max), gate_width_min)
        r = _tuning(omega_i, omega_k, gate_w)
        s = _spatial_overlap_from_anchors(
            pos_x,
            pos_y,
            pos_z,
            particle_pos_ptr,
            anchor_idx_ptr,
            anchor_weight_ptr,
            k.to(tl.int32),
            domain_x,
            domain_y,
            domain_z,
            spatial_sigma,
            N=N,
        )
        t = r * s
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
    particle_pos_ptr,  # fp32 [N,3]
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
    domain_x: tl.float32,
    domain_y: tl.float32,
    domain_z: tl.float32,
    spatial_sigma: tl.float32,
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
    pos_x = tl.load(particle_pos_ptr + i * 3 + 0)
    pos_y = tl.load(particle_pos_ptr + i * 3 + 1)
    pos_z = tl.load(particle_pos_ptr + i * 3 + 2)

    num_car = tl.load(num_carriers_ptr).to(tl.int32)
    total = tl.zeros((), tl.float32)
    for k in range(0, max_carriers):
        active = k < num_car
        omega_k = tl.load(carrier_omega_ptr + k, mask=active, other=0.0)
        gate_w = tl.load(carrier_gate_ptr + k, mask=active, other=1.0)
        gate_w = tl.maximum(tl.minimum(gate_w, gate_width_max), gate_width_min)
        r = _tuning(omega_i, omega_k, gate_w)
        s = _spatial_overlap_from_anchors(
            pos_x,
            pos_y,
            pos_z,
            particle_pos_ptr,
            anchor_idx_ptr,
            anchor_weight_ptr,
            k.to(tl.int32),
            domain_x,
            domain_y,
            domain_z,
            spatial_sigma,
            N=N,
        )
        total += (r * s)
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
    particle_pos: torch.Tensor,
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
    domain_x: float,
    domain_y: float,
    domain_z: float,
    spatial_sigma: float,
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
    w_omega2_sum = torch.zeros((current_carriers,), device=osc_phase.device, dtype=torch.float32)
    w_amp_sum = torch.zeros((current_carriers,), device=osc_phase.device, dtype=torch.float32)

    # Pass 1: accumulate block partials into per-carrier sums
    carrier_block_accum_kernel[(current_carriers, blocks)](
        osc_phase,
        osc_omega,
        osc_amp,
        particle_pos,
        carrier_omega,
        carrier_gate_width,
        anchor_idx,
        anchor_weight,
        force_r,
        force_i,
        w_sum,
        w_omega_sum,
        w_omega2_sum,
        w_amp_sum,
        N=N,
        BLOCK_N=BLOCK_N,
        offender_weight_floor=float(params.offender_weight_floor),
        gate_width_min=float(params.gate_width_min),
        gate_width_max=float(params.gate_width_max),
        domain_x=float(domain_x),
        domain_y=float(domain_y),
        domain_z=float(domain_z),
        spatial_sigma=float(spatial_sigma),
        num_warps=1,
    )

    # Pass 2: finalize updates + split + crystallization + anchor refresh
    carrier_finalize_and_split_kernel[(current_carriers,)](
        osc_phase,
        osc_omega,
        osc_amp,
        particle_pos,
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
        w_omega2_sum,
        w_amp_sum,
        float(domain_x),
        float(domain_y),
        float(domain_z),
        float(spatial_sigma),
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


@triton.jit
def carrier_gpe_step_kernel(
    # coherence field Ψ(ω)
    carrier_real_ptr,
    carrier_imag_ptr,
    carrier_omega_ptr,
    carrier_gate_ptr,
    # anchors (for spatial overlap approximation)
    anchor_idx_ptr,
    anchor_phase_ptr,
    anchor_weight_ptr,
    # oscillator state (for anchor refresh)
    osc_phase_ptr,
    osc_omega_ptr,
    osc_amp_ptr,
    particle_pos_ptr,  # fp32 [N,3]
    # observations
    w_sum_ptr,
    domain_x: tl.float32,
    domain_y: tl.float32,
    domain_z: tl.float32,
    spatial_sigma: tl.float32,
    N: tl.constexpr,
    current_carriers: tl.constexpr,
    # GPE params
    dt: tl.constexpr,
    hbar_eff: tl.constexpr,
    mass_eff: tl.constexpr,
    g_interaction: tl.constexpr,
    energy_decay: tl.constexpr,
    chemical_potential: tl.constexpr,
    inv_domega2: tl.constexpr,
    # coupling params
    gate_width_min: tl.constexpr,
    gate_width_max: tl.constexpr,
    # anchor refresh
    anchor_eps: tl.constexpr,
    rng_seed: tl.constexpr,
):
    k = tl.program_id(0)

    cr = tl.load(carrier_real_ptr + k)
    ci = tl.load(carrier_imag_ptr + k)
    omega_k = tl.load(carrier_omega_ptr + k)
    gate_w = tl.load(carrier_gate_ptr + k)
    gate_w = tl.maximum(tl.minimum(gate_w, gate_width_max), gate_width_min)

    w_sum = tl.load(w_sum_ptr + k)
    V_ext = -w_sum

    # Strang-style split-step (local potential/nonlinear ↔ kinetic)
    #
    # This is an open system (observations + optional dissipation), but the symmetric
    # split substantially improves phase fidelity vs blending everything in one step.
    hb = tl.maximum(hbar_eff, 1e-8)
    half_dt = 0.5 * dt

    # half-step potential/nonlinear rotation at k
    dens = cr * cr + ci * ci
    H = V_ext + g_interaction * dens - chemical_potential
    theta = -(H * half_dt) / hb
    rot_r = tl.cos(theta)
    rot_i = tl.sin(theta)
    cr2 = cr * rot_r - ci * rot_i
    ci2 = cr * rot_i + ci * rot_r
    cr = cr2
    ci = ci2

    # kinetic / tunneling: i*(ħ/2m) ∇²ψ
    if mass_eff > 0.0 and inv_domega2 > 0.0:
        left = tl.maximum(k - 1, 0)
        right = tl.minimum(k + 1, current_carriers - 1)

        # Load neighbors and apply the same half-step potential rotation locally
        # so the Laplacian matches the split-step ordering.
        cr_l = tl.load(carrier_real_ptr + left)
        ci_l = tl.load(carrier_imag_ptr + left)
        cr_r = tl.load(carrier_real_ptr + right)
        ci_r = tl.load(carrier_imag_ptr + right)

        w_sum_l = tl.load(w_sum_ptr + left)
        V_ext_l = -w_sum_l
        dens_l = cr_l * cr_l + ci_l * ci_l
        H_l = V_ext_l + g_interaction * dens_l - chemical_potential
        theta_l = -(H_l * half_dt) / hb
        rot_lr = tl.cos(theta_l)
        rot_li = tl.sin(theta_l)
        cr_l2 = cr_l * rot_lr - ci_l * rot_li
        ci_l2 = cr_l * rot_li + ci_l * rot_lr
        cr_l = cr_l2
        ci_l = ci_l2

        w_sum_r = tl.load(w_sum_ptr + right)
        V_ext_r = -w_sum_r
        dens_r = cr_r * cr_r + ci_r * ci_r
        H_r = V_ext_r + g_interaction * dens_r - chemical_potential
        theta_r = -(H_r * half_dt) / hb
        rot_rr = tl.cos(theta_r)
        rot_ri = tl.sin(theta_r)
        cr_r2 = cr_r * rot_rr - ci_r * rot_ri
        ci_r2 = cr_r * rot_ri + ci_r * rot_rr
        cr_r = cr_r2
        ci_r = ci_r2

        lap_r = cr_l - 2.0 * cr + cr_r
        lap_i = ci_l - 2.0 * ci + ci_r
        kin = (hb * dt) / (2.0 * mass_eff)
        # i * lap = (-lap_i, lap_r)
        cr = cr + (-lap_i) * (kin * inv_domega2)
        ci = ci + (lap_r) * (kin * inv_domega2)

    # second half-step potential/nonlinear rotation at k (recompute density after kinetic)
    dens = cr * cr + ci * ci
    H = V_ext + g_interaction * dens - chemical_potential
    theta = -(H * half_dt) / hb
    rot_r = tl.cos(theta)
    rot_i = tl.sin(theta)
    cr2 = cr * rot_r - ci * rot_i
    ci2 = cr * rot_i + ci * rot_r
    cr = cr2
    ci = ci2

    # dissipation
    if energy_decay > 0.0:
        s = tl.exp(-(energy_decay * dt))
        cr *= s
        ci *= s

    # anchor refresh (ε-greedy)
    if anchor_eps > 0.0 and N > 0:
        h = _hash_u32((rng_seed ^ (k.to(tl.uint32) * 0xB4B82E39) ^ 0x1C3A5F7D).to(tl.uint32))
        u = _u01_from_u32(h)
        do = u <= anchor_eps
        slot = (_hash_u32(h + 0x3C6EF372) % ANCHORS).to(tl.int32)
        base = k * ANCHORS + slot
        chosen = (_hash_u32(h + 0x9E3779B9) % N).to(tl.int32)

        phi_i = tl.load(osc_phase_ptr + chosen, mask=do, other=0.0)
        omega_i = tl.load(osc_omega_ptr + chosen, mask=do, other=0.0)
        amp_i = tl.load(osc_amp_ptr + chosen, mask=do, other=0.0)

        pos_x = tl.load(particle_pos_ptr + chosen * 3 + 0, mask=do, other=0.0)
        pos_y = tl.load(particle_pos_ptr + chosen * 3 + 1, mask=do, other=0.0)
        pos_z = tl.load(particle_pos_ptr + chosen * 3 + 2, mask=do, other=0.0)

        r = _tuning(omega_i, omega_k, gate_w)
        s_ov = _spatial_overlap_from_anchors(
            pos_x,
            pos_y,
            pos_z,
            particle_pos_ptr,
            anchor_idx_ptr,
            anchor_weight_ptr,
            k.to(tl.int32),
            domain_x,
            domain_y,
            domain_z,
            spatial_sigma,
            N=N,
        )
        w = (r * s_ov) * amp_i
        psi_phase = tl.atan2(ci, cr)
        dphi = _wrap_pi(phi_i - psi_phase)

        tl.store(anchor_idx_ptr + base, chosen, mask=do)
        tl.store(anchor_phase_ptr + base, dphi, mask=do)
        tl.store(anchor_weight_ptr + base, w, mask=do)

    tl.store(carrier_real_ptr + k, cr)
    tl.store(carrier_imag_ptr + k, ci)


def coherence_gpe_step(
    *,
    osc_phase: torch.Tensor,
    osc_omega: torch.Tensor,
    osc_amp: torch.Tensor,
    particle_pos: torch.Tensor,
    carrier_real: torch.Tensor,
    carrier_imag: torch.Tensor,
    carrier_omega: torch.Tensor,
    carrier_gate_width: torch.Tensor,
    anchor_idx: torch.Tensor,
    anchor_phase: torch.Tensor,
    anchor_weight: torch.Tensor,
    current_carriers: int,
    # GPE params
    dt: float,
    hbar_eff: float,
    mass_eff: float,
    g_interaction: float,
    energy_decay: float,
    chemical_potential: float,
    inv_domega2: float,
    anchor_eps: float,
    rng_seed: int,
    # coupling params
    offender_weight_floor: float,
    gate_width_min: float,
    gate_width_max: float,
    domain_x: float,
    domain_y: float,
    domain_z: float,
    spatial_sigma: float,
) -> None:
    _require_triton()
    assert osc_phase.is_cuda
    N = int(osc_phase.numel())
    if N == 0 or current_carriers == 0:
        return

    # Accumulators (per active carrier)
    force_r = torch.zeros((current_carriers,), device=osc_phase.device, dtype=torch.float32)
    force_i = torch.zeros((current_carriers,), device=osc_phase.device, dtype=torch.float32)
    w_sum = torch.zeros((current_carriers,), device=osc_phase.device, dtype=torch.float32)
    w_omega_sum = torch.zeros((current_carriers,), device=osc_phase.device, dtype=torch.float32)
    w_omega2_sum = torch.zeros((current_carriers,), device=osc_phase.device, dtype=torch.float32)
    w_amp_sum = torch.zeros((current_carriers,), device=osc_phase.device, dtype=torch.float32)

    BLOCK_N = 256
    blocks = triton.cdiv(N, BLOCK_N)
    carrier_block_accum_kernel[(current_carriers, blocks)](
        osc_phase,
        osc_omega,
        osc_amp,
        particle_pos,
        carrier_omega,
        carrier_gate_width,
        anchor_idx,
        anchor_weight,
        force_r,
        force_i,
        w_sum,
        w_omega_sum,
        w_omega2_sum,
        w_amp_sum,
        N=N,
        BLOCK_N=BLOCK_N,
        offender_weight_floor=float(offender_weight_floor),
        gate_width_min=float(gate_width_min),
        gate_width_max=float(gate_width_max),
        domain_x=float(domain_x),
        domain_y=float(domain_y),
        domain_z=float(domain_z),
        spatial_sigma=float(spatial_sigma),
        num_warps=1,
    )

    carrier_gpe_step_kernel[(current_carriers,)](
        carrier_real,
        carrier_imag,
        carrier_omega,
        carrier_gate_width,
        anchor_idx,
        anchor_phase,
        anchor_weight,
        osc_phase,
        osc_omega,
        osc_amp,
        particle_pos,
        w_sum,
        float(domain_x),
        float(domain_y),
        float(domain_z),
        float(spatial_sigma),
        N=N,
        current_carriers=int(current_carriers),
        dt=float(dt),
        hbar_eff=float(hbar_eff),
        mass_eff=float(mass_eff),
        g_interaction=float(g_interaction),
        energy_decay=float(energy_decay),
        chemical_potential=float(chemical_potential),
        inv_domega2=float(inv_domega2),
        gate_width_min=float(gate_width_min),
        gate_width_max=float(gate_width_max),
        anchor_eps=float(anchor_eps),
        rng_seed=int(rng_seed) & 0xFFFFFFFF,
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


def coherence_topdown_bias_energies(**kwargs) -> None:
    """Alias for `topdown_bias_energies` (coherence terminology)."""
    return topdown_bias_energies(**kwargs)


def update_oscillator_phases(
    *,
    osc_phase: torch.Tensor,
    osc_omega: torch.Tensor,
    osc_amp: torch.Tensor,
    particle_pos: torch.Tensor,
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
    domain_x: float,
    domain_y: float,
    domain_z: float,
    spatial_sigma: float,
) -> None:
    _require_triton()
    if N == 0:
        return
    grid = (triton.cdiv(N, 1),)
    spectral_update_oscillator_phases_kernel[grid](
        osc_phase,
        osc_omega,
        osc_amp,
        particle_pos,
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
        float(domain_x),
        float(domain_y),
        float(domain_z),
        float(spatial_sigma),
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


def coherence_update_oscillator_phases(**kwargs) -> None:
    """Alias for `update_oscillator_phases` (coherence terminology)."""
    return update_oscillator_phases(**kwargs)


def spawn_uncoupled(
    *,
    osc_phase: torch.Tensor,
    osc_omega: torch.Tensor,
    osc_amp: torch.Tensor,
    particle_pos: torch.Tensor,
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
    domain_x: float,
    domain_y: float,
    domain_z: float,
    spatial_sigma: float,
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
        particle_pos,
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
        float(domain_x),
        float(domain_y),
        float(domain_z),
        float(spatial_sigma),
        N=N,
        max_carriers=max_carriers,
        coupling_threshold=float(coupling_threshold),
        gate_width_init=float(gate_width_init),
        gate_width_min=float(gate_width_min),
        gate_width_max=float(gate_width_max),
        num_warps=1,
    )


def coherence_spawn_uncoupled(**kwargs) -> None:
    """Alias for `spawn_uncoupled` (coherence terminology)."""
    return spawn_uncoupled(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS-BASED PARAMETER ADAPTER
# ═══════════════════════════════════════════════════════════════════════════════
#
# This section provides an adapter to convert physics-derived ModeParams
# into the format expected by the existing kernels (SpectralParams).
#
# The mapping ensures that:
# - Arbitrary thresholds → physics-derived visibility ratios
# - Multiple decay multipliers → single Lindblad decay
# - Fixed crystallize_age → coherence_steps from τ_coh/dt
# - Gate width bounds → uncertainty principle limits
# ═══════════════════════════════════════════════════════════════════════════════


def spectral_params_from_mode_physics(params: ModeParams) -> SpectralParams:
    """Convert physics-derived ModeParams to legacy SpectralParams format.

    [PURPOSE] Bridge between new physics-based system and existing kernels.
    [DESIGN] All "arbitrary" parameters are now computed from physics.

    Parameter derivations:
    - gate_width_min/max: from uncertainty principle & thermal bandwidth
    - carrier_reg: from Lindblad decay rate
    - decay multipliers: from exp(-Γ·dt) where Γ is decay_rate
    - stable_amp_threshold: visibility > 1.0 (above thermal noise)
    - crystallize_amp_threshold: visibility > 2.0 (well above noise)
    - crystallize_age: from coherence_steps (τ_coh / dt)
    """
    # Decay factor from Lindblad rate
    decay = math.exp(-params.decay_rate * params.dt) if params.decay_rate * params.dt < 700 else 0.0

    return SpectralParams(
        # Core parameters (direct from ModeParams)
        dt=params.dt,
        temperature=params.temperature,
        rng_seed=params.rng_seed,
        coupling_scale=params.coupling_scale,

        # [DERIVED] Dissipation from Lindblad formula
        # [FORMULA] carrier_reg = Γ = γ · k_B T / ℏ
        carrier_reg=params.decay_rate,

        # [DERIVED] Bandwidth from uncertainty principle
        # [FORMULA] Δω_min = 1/(2·dt), Δω_max = k_B T / ℏ
        gate_width_min=params.bandwidth_min,
        gate_width_max=params.bandwidth_max,

        # [DERIVED] Decay multipliers from Lindblad
        # [FORMULA] decay = exp(-Γ·dt)
        # [NOTES] Excited/coherent modes decay; condensed modes do not
        volatile_decay_mul=decay,
        stable_decay_mul=decay,
        crystallized_decay_mul=1.0,  # Condensed modes don't decay

        # [DERIVED] Amplitude thresholds from visibility ratio
        # [FORMULA] visibility = A / A_thermal where A_thermal = noise_floor_scale / √ω
        # [NOTES] For threshold, we use noise_floor_scale directly (assumes ω ≈ 1)
        #         visible if A > noise_floor_scale → threshold = noise_floor_scale
        #         strongly visible if A > 2 × noise_floor_scale
        stable_amp_threshold=params.noise_floor_scale,      # visibility > 1
        crystallize_amp_threshold=2.0 * params.noise_floor_scale,  # visibility > 2

        # [DERIVED] Crystallization age from coherence time
        # [FORMULA] crystallize_age = τ_coh / dt = ℏ/(k_B T) / dt
        crystallize_age=params.coherence_steps,

        # [CHOICE] Conflict threshold — kept as reasonable default
        # [REASON] coherence = 1 - conflict; conflict < 0.5 means > 50% phase coherence
        # [NOTES] This could be derived from phase variance, but 0.5 is sensible
        conflict_threshold=0.5,
        crystallize_conflict_threshold=0.2,  # High coherence for condensation

        # [CHOICE] Weight floor for coupling calculation
        # [REASON] Prevents numerical issues with near-zero weights
        offender_weight_floor=1e-3,

        # [CHOICE] EMA alpha for conflict smoothing
        # [FORMULA] α = dt / (τ + dt) where τ is smoothing time
        # [NOTES] Using τ ≈ 10×dt gives α ≈ 0.1
        ema_alpha=0.1,
        recenter_alpha=0.1,

        # Mode (algorithm behavior, not physics)
        mode=params.mode,

        # [CHOICE] Top-down coupling from condensed modes
        # [REASON] Condensed modes bias oscillators toward learned patterns
        # [NOTES] Scale relative to coupling strength
        crystallized_coupling_boost=1.0,
        topdown_phase_scale=params.coupling_scale * 0.5,
        topdown_energy_scale=params.coupling_scale * 0.5,
        topdown_random_energy_eps=0.02,

        # [CHOICE] Anchor refresh randomness
        anchor_random_eps=0.05,

        # [CHOICE] Frequency repulsion for disambiguation
        # [REASON] Prevents modes from collapsing to same frequency
        repulsion_scale=0.05,
    )


def coherence_params_from_mode_physics(params: ModeParams) -> CoherenceParams:
    """Alias for `spectral_params_from_mode_physics`.

    The underlying kernels historically used "spectral" naming. The current model
    is a coherence field Ψ(ω), so prefer the coherence_* names in new code.
    """
    return spectral_params_from_mode_physics(params)


def mode_update_with_physics(
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
    current_modes: int,
    max_modes: int,
    params: ModeParams,
) -> None:
    """Mode update using physics-derived parameters.

    This is the recommended interface for new code. It takes ModeParams
    (physics-derived) and calls the existing kernel with appropriate values.
    """
    legacy_params = spectral_params_from_mode_physics(params)
    carrier_update_and_split(
        osc_phase=osc_phase,
        osc_omega=osc_omega,
        osc_amp=osc_amp,
        carrier_real=carrier_real,
        carrier_imag=carrier_imag,
        carrier_omega=carrier_omega,
        carrier_gate_width=carrier_gate_width,
        carrier_conflict=carrier_conflict,
        carrier_state=carrier_state,
        carrier_age=carrier_age,
        anchor_idx=anchor_idx,
        anchor_phase=anchor_phase,
        anchor_weight=anchor_weight,
        num_carriers=num_carriers,
        spawned_from=spawned_from,
        random_phases=random_phases,
        energy_stats=energy_stats,
        current_carriers=current_modes,
        max_carriers=max_modes,
        params=legacy_params,
    )

