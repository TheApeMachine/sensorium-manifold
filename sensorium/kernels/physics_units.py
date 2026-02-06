"""Unit systems + physical constants (auditable, non-tuneable)

This module defines:
- CODATA physical constants in SI units (universal constants).
- An explicit base-unit mapping from simulation units -> SI units.
- Deterministic conversion of physical constants into simulation units.

This module is used to define the physical constants and unit systems for the simulation.
It is used to convert the physical constants and unit systems from SI to simulation units,
and vice versa.

These are to be considered non-tunable constants, and should not be exploited as
hyperparameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SIConstants:
    """CODATA / defined SI constants (numerical values in SI units).

    References:
    - k_B is exact by definition in SI.
    - G uses CODATA recommended value (with measurement uncertainty).
    - σ uses CODATA value (derived from other constants).
    """

    # [CHOICE] Newtonian gravitational constant (SI)
    # [FORMULA] G_SI ≈ 6.67430e-11 m^3 kg^-1 s^-2
    # [REASON] universal constant (measured)
    # [NOTES] if we want uncertainty propagation, we can add ± here later.
    G: float = 6.67430e-11

    # [CHOICE] Boltzmann constant (SI)
    # [FORMULA] k_B = 1.380649e-23 J K^-1 (exact)
    # [REASON] universal constant (defined)
    k_B: float = 1.380649e-23

    # [CHOICE] Avogadro constant (SI)
    # [FORMULA] N_A = 6.02214076e23 mol^-1 (exact)
    # [REASON] defines the mole in SI; needed to derive R = N_A k_B
    N_A: float = 6.02214076e23

    # [CHOICE] Stefan–Boltzmann constant (SI)
    # [FORMULA] σ ≈ 5.670374419e-8 W m^-2 K^-4
    # [REASON] universal constant (derived from fundamental constants)
    sigma_SB: float = 5.670374419e-8

    # [CHOICE] Planck constant (SI)
    # [FORMULA] h = 6.62607015e-34 J s (exact)
    # [REASON] universal constant (defined); needed for quantum oscillator equilibrium
    # [NOTES] We use ℏ = h / (2π) for the Planck oscillator mean energy.
    h: float = 6.62607015e-34


@dataclass(frozen=True)
class UnitSystem:
    """Mapping from simulation base units to SI base units.

    Interpretation:
    - 1 simulation length unit = length_unit_m meters
    - 1 simulation mass unit   = mass_unit_kg kilograms
    - 1 simulation time unit   = time_unit_s seconds
    - 1 simulation temperature = temperature_unit_K kelvin

    This is not a tunable parameter: it is a declaration of what your simulation
    units *mean* in SI.
    """

    # [CHOICE] base unit mapping (sim → SI)
    # [FORMULA] x_SI = x_sim * length_unit_m, etc.
    # [REASON] makes dimensional analysis explicit and auditable
    # [NOTES] choose these to match the real physical scale being simulated.
    length_unit_m: float
    mass_unit_kg: float
    time_unit_s: float
    temperature_unit_K: float

    name: str = "custom"

    @staticmethod
    def si(*, name: str = "si") -> "UnitSystem":
        """Identity mapping: 1 sim unit == 1 SI unit."""
        return UnitSystem(
            length_unit_m=1.0,
            mass_unit_kg=1.0,
            time_unit_s=1.0,
            temperature_unit_K=1.0,
            name=name,
        )

    @staticmethod
    def omega_natural(
        *,
        gamma: float,
        molecular_weight_kg_per_mol: float,
        length_unit_m: float = 1.0,
        name: str = "omega_natural",
        si: SIConstants | None = None,
    ) -> "UnitSystem":
        """Natural unit system consistent with ω∈O(1) and (ħ=k_B=1).

        Goal (in *simulation units*):
        - ħ_sim = 1
        - k_B_sim = 1
        - c_v = 1  ⇔  R_specific_sim = (γ - 1)

        This makes the Planck crossover ω≈T meaningful when ω∈[0,2) and T∼O(1).

        Construction:
        - Use a fixed length unit (default 1 meter) as a declaration, not a knob.
        - Derive mass/time/temperature units deterministically from CODATA + medium:
            mass_unit_kg = (γ-1) * k_B_SI / R_specific_SI
            time_unit_s   = (L^2 * mass_unit_kg) / ħ_SI
            temp_unit_K   = ħ_SI^2 / (k_B_SI * L^2 * mass_unit_kg)

        where R_specific_SI = (N_A k_B)/M_molar  [J/(kg K)], and ħ_SI = h/(2π).
        """
        si = si or SIConstants()
        g = float(gamma)
        if not (g > 1.0):
            raise ValueError(f"gamma must be > 1, got {gamma}")
        if not (molecular_weight_kg_per_mol > 0.0):
            raise ValueError("molecular_weight_kg_per_mol must be > 0")
        L = float(length_unit_m)
        if not (L > 0.0):
            raise ValueError("length_unit_m must be > 0")

        # R_specific in SI (J/(kg K)).
        R_universal_si = float(si.N_A) * float(si.k_B)  # J/(mol K)
        R_specific_si = R_universal_si / float(molecular_weight_kg_per_mol)  # J/(kg K)
        if not (R_specific_si > 0.0) or not math.isfinite(R_specific_si):
            raise ValueError("Non-finite R_specific_si")

        hbar_si = float(si.h) / (2.0 * math.pi)
        if not (hbar_si > 0.0) or not math.isfinite(hbar_si):
            raise ValueError("Non-finite ħ")

        # Derived units.
        mass_unit_kg = (g - 1.0) * float(si.k_B) / float(R_specific_si)
        time_unit_s = (L * L * mass_unit_kg) / float(hbar_si)
        temperature_unit_K = (hbar_si * hbar_si) / (float(si.k_B) * L * L * mass_unit_kg)

        return UnitSystem(
            length_unit_m=L,
            mass_unit_kg=float(mass_unit_kg),
            time_unit_s=float(time_unit_s),
            temperature_unit_K=float(temperature_unit_K),
            name=name,
        )


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants expressed in *simulation units* for a given UnitSystem."""

    # [CHOICE] constant source tag
    # [FORMULA] N/A
    # [REASON] allows auditing the provenance of values
    source: Literal["codata_si_derived"]

    G: float
    k_B: float
    sigma_SB: float
    hbar: float

    @staticmethod
    def _to_sim(value_si: float, *, L: float, M: float, T: float, K: float, dims: tuple[int, int, int, int]) -> float:
        """Convert an SI quantity into simulation units via base-unit exponents.

        dims = (a, b, c, d) corresponds to units m^a kg^b s^c K^d.

        If 1 sim unit = U_SI in SI units, then:
          1 SI unit = 1/U_sim in sim units,
        so:
          value_sim = value_SI * Π (U_SI)^(-exponent)
        """
        a, b, c, d = dims
        return float(value_si) * (L ** (-a)) * (M ** (-b)) * (T ** (-c)) * (K ** (-d))

    @classmethod
    def from_codata_si(cls, units: UnitSystem, si: SIConstants | None = None) -> "PhysicalConstants":
        si = si or SIConstants()
        L = float(units.length_unit_m)
        M = float(units.mass_unit_kg)
        T = float(units.time_unit_s)
        K = float(units.temperature_unit_K)

        # Dimensions:
        # - G: m^3 kg^-1 s^-2
        # - k_B: J/K = (kg m^2 s^-2) K^-1
        # - sigma: W m^-2 K^-4 = (kg s^-3) K^-4  (since W=kg m^2 s^-3)
        # - hbar: J s = (kg m^2 s^-1)
        G_sim = cls._to_sim(si.G, L=L, M=M, T=T, K=K, dims=(3, -1, -2, 0))
        kB_sim = cls._to_sim(si.k_B, L=L, M=M, T=T, K=K, dims=(2, 1, -2, -1))
        sigma_sim = cls._to_sim(si.sigma_SB, L=L, M=M, T=T, K=K, dims=(0, 1, -3, -4))
        hbar_si = float(si.h) / (2.0 * math.pi)
        hbar_sim = cls._to_sim(hbar_si, L=L, M=M, T=T, K=K, dims=(2, 1, -1, 0))

        return cls(source="codata_si_derived", G=G_sim, k_B=kB_sim, sigma_SB=sigma_sim, hbar=hbar_sim)


def gas_R_specific_sim(units: UnitSystem, *, molecular_weight_kg_per_mol: float, si: SIConstants | None = None) -> float:
    """Compute the ideal-gas specific gas constant R_specific in simulation units.

    [CHOICE] ideal gas constant derivation
    [FORMULA] R = N_A k_B,  R_specific = R / M
    [REASON] ideal-gas EOS uses p = ρ R_specific T (mass-based continuum form)
    [NOTES] `molecular_weight_kg_per_mol` is the molar mass M in SI units (kg/mol).
            The mole is not a simulation base unit, so we compute R_specific in SI
            first (J/(kg K)), then convert into simulation units via (m^2 s^-2 K^-1).
    """
    if not (molecular_weight_kg_per_mol > 0.0):
        raise ValueError(f"molecular_weight_kg_per_mol must be > 0, got {molecular_weight_kg_per_mol}")

    si = si or SIConstants()
    R_universal_si = float(si.N_A) * float(si.k_B)  # J/(mol K)
    R_specific_si = R_universal_si / float(molecular_weight_kg_per_mol)  # J/(kg K)

    # J/(kg K) = (m^2 s^-2) K^-1  => dims=(2, 0, -2, -1)
    return PhysicalConstants._to_sim(
        R_specific_si,
        L=float(units.length_unit_m),
        M=float(units.mass_unit_kg),
        T=float(units.time_unit_s),
        K=float(units.temperature_unit_K),
        dims=(2, 0, -2, -1),
    )


def dynamic_viscosity_sim(units: UnitSystem, *, mu_si_pa_s: float) -> float:
    """Convert dynamic viscosity μ from SI (Pa·s) to simulation units.

    SI units: Pa·s = kg·m^-1·s^-1  => dims = (-1, 1, -1, 0)
    """
    if not (mu_si_pa_s > 0.0) or not math.isfinite(float(mu_si_pa_s)):
        return 0.0
    return PhysicalConstants._to_sim(
        float(mu_si_pa_s),
        L=float(units.length_unit_m),
        M=float(units.mass_unit_kg),
        T=float(units.time_unit_s),
        K=float(units.temperature_unit_K),
        dims=(-1, 1, -1, 0),
    )


def assert_finite_constants(c: PhysicalConstants) -> None:
    """Fail loudly if constants are non-finite."""
    for name, v in (("G", c.G), ("k_B", c.k_B), ("sigma_SB", c.sigma_SB), ("hbar", c.hbar)):
        if not math.isfinite(float(v)):
            raise ValueError(f"Non-finite constant {name}={v!r} (source={c.source})")


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM THERMODYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════
#
# These functions compute quantities for the spectral (quantum) domain.
# All thresholds and timescales derive from temperature T, frequency ω,
# and physical constants. No arbitrary parameters.
#
# The geometric domain provides T. The spectral domain uses T to determine
# which modes are visible above thermal noise, how fast phases decohere,
# and when condensation (crystallization) occurs.
# ═══════════════════════════════════════════════════════════════════════════════


def bose_einstein_occupation(omega: float, T: float, c: PhysicalConstants) -> float:
    """Mean occupation number for a quantum harmonic oscillator.

    [CHOICE] Bose-Einstein distribution
    [FORMULA] ⟨n⟩ = 1 / (exp(ℏω/k_B T) - 1)
    [REASON] Fundamental result of quantum statistical mechanics for bosonic modes.
    [NOTES] Returns 0 if T ≤ 0 or ω ≤ 0. For ℏω >> k_B T, occupation → 0 (frozen out).
            For ℏω << k_B T, occupation → k_B T / ℏω (classical limit).
    """
    if T <= 0.0 or omega <= 0.0:
        return 0.0
    x = (c.hbar * omega) / (c.k_B * T)
    if x > 700.0:  # prevent overflow in exp
        return 0.0
    return 1.0 / (math.exp(x) - 1.0)


def thermal_amplitude(omega: float, T: float, c: PhysicalConstants, m: float = 1.0) -> float:
    """RMS amplitude of thermal fluctuations for a harmonic oscillator.

    [CHOICE] Zero-point + thermal fluctuation amplitude
    [FORMULA] A_thermal = √((ℏ/2mω) · (2⟨n⟩ + 1))
    [REASON] This is the quantum mechanical expectation value ⟨x²⟩^(1/2).
             Even at T=0, there are zero-point fluctuations (⟨n⟩=0 gives A=√(ℏ/2mω)).
             A mode is "visible" only if its amplitude exceeds this thermal floor.
    [NOTES] m is effective mass (default 1 in natural units).
            The noise floor rises with temperature and falls with frequency.
    """
    if omega <= 0.0 or m <= 0.0:
        return 0.0
    n = bose_einstein_occupation(omega, T, c)
    # A² = (ℏ/2mω)(2n + 1)
    return math.sqrt((c.hbar / (2.0 * m * omega)) * (2.0 * n + 1.0))


def thermal_coherence_time(T: float, c: PhysicalConstants) -> float:
    """Characteristic time for phase decoherence due to thermal fluctuations.

    [CHOICE] Thermal coherence time
    [FORMULA] τ_coh = ℏ / (k_B T)
    [REASON] Dimensional analysis: ℏ/kT is the only timescale from these quantities.
             Physically, thermal energy kT causes phase to diffuse on this timescale.
             At T=300K, τ_coh ≈ 25 femtoseconds. For our simulation units, scale accordingly.
    [NOTES] Returns infinity if T ≤ 0 (no thermal decoherence at absolute zero).
    """
    if T <= 0.0:
        return float('inf')
    return c.hbar / (c.k_B * T)


def thermal_frequency(T: float, c: PhysicalConstants) -> float:
    """Characteristic frequency of thermal fluctuations.

    [CHOICE] Thermal frequency (inverse of coherence time)
    [FORMULA] ω_thermal = k_B T / ℏ
    [REASON] Modes with ω >> ω_thermal are frozen out (quantum regime).
             Modes with ω << ω_thermal behave classically.
             This is the natural bandwidth scale for thermal processes.
    [NOTES] Returns 0 if T ≤ 0.
    """
    if T <= 0.0:
        return 0.0
    return (c.k_B * T) / c.hbar


def uncertainty_bandwidth(dt: float) -> float:
    """Minimum resolvable frequency bandwidth from uncertainty principle.

    [CHOICE] Heisenberg uncertainty for frequency measurement
    [FORMULA] Δω_min = 1 / (2 · dt)
    [REASON] Energy-time uncertainty ΔE·Δt ≥ ℏ/2 implies Δω·Δt ≥ 1/2.
             We cannot resolve frequency differences smaller than this.
    [NOTES] This sets the minimum linewidth (γ) for mode frequency selectivity.
    """
    if dt <= 0.0:
        return float('inf')
    return 0.5 / dt


def lindblad_decay_rate(T: float, coupling: float, c: PhysicalConstants) -> float:
    """Decay rate for a mode coupled to a thermal bath (Lindblad dissipation).

    [CHOICE] Thermal relaxation rate from open quantum systems
    [FORMULA] Γ = γ · (k_B T / ℏ) = γ · ω_thermal
    [REASON] The Lindblad master equation for a harmonic oscillator coupled to
             a thermal bath gives decay rate proportional to temperature.
             γ is the dimensionless coupling strength to the bath.
    [NOTES] decay_factor = exp(-Γ · dt) per timestep.
            Stronger coupling (larger γ) → faster thermalization.
    """
    omega_th = thermal_frequency(T, c)
    return coupling * omega_th


def mode_visibility_ratio(amplitude: float, omega: float, T: float, c: PhysicalConstants, m: float = 1.0) -> float:
    """Ratio of mode amplitude to thermal noise floor.

    [CHOICE] Signal-to-noise ratio for mode detection
    [FORMULA] visibility = A / A_thermal
    [REASON] A mode is:
             - THERMAL (noise) if visibility ≤ 1
             - EXCITED if visibility > 1
             - STRONGLY EXCITED if visibility >> 1
    [NOTES] This replaces arbitrary amplitude thresholds with physics.
    """
    a_th = thermal_amplitude(omega, T, c, m)
    if a_th <= 0.0:
        return float('inf') if amplitude > 0.0 else 0.0
    return amplitude / a_th


@dataclass(frozen=True)
class QuantumThermodynamicState:
    """Precomputed quantum thermodynamic quantities for a given temperature.

    Compute once per timestep (when T changes), then use for all mode calculations.
    This avoids recomputing expensive transcendentals for each mode.
    """
    T: float                    # Temperature (from geometric domain)
    omega_thermal: float        # k_B T / ℏ — characteristic thermal frequency
    tau_coherence: float        # ℏ / k_B T — phase decoherence timescale

    @classmethod
    def from_temperature(cls, T: float, c: PhysicalConstants) -> "QuantumThermodynamicState":
        """Compute quantum thermodynamic state from temperature."""
        return cls(
            T=T,
            omega_thermal=thermal_frequency(T, c),
            tau_coherence=thermal_coherence_time(T, c),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODE PHYSICS PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
#
# All quantities are derived from physics — no tunable hyperparameters.
#
# The ONLY inputs are:
#   - T: temperature from thermodynamic domain
#   - dt: simulation timestep
#   - c: physical constants
#   - bath_coupling: dimensionless coupling to thermal bath (has physical meaning)
#
# Everything else is computed.
# ═══════════════════════════════════════════════════════════════════════════════


# Mode states (physics-based definitions)
MODE_THERMAL = 0      # Amplitude ≤ noise floor (not explicitly tracked)
MODE_EXCITED = 1      # Amplitude > noise floor, phase incoherent
MODE_COHERENT = 2     # Amplitude > noise floor, phase stable
MODE_CONDENSED = 3    # Macroscopic occupation, frozen (BEC-like)


@dataclass(frozen=True)
class ModePhysics:
    """Physics-derived parameters for coherence mode dynamics.

    All values are computed from temperature T, timestep dt, and physical constants.
    No arbitrary hyperparameters.

    [DESIGN PRINCIPLE]
    The thermodynamic domain computes T. This class takes T and derives everything
    the coherence domain needs: noise floors, decay rates, bandwidths, thresholds.
    """

    # ─── Inputs (from simulation state) ───
    T: float                    # Temperature from thermodynamic domain [K or sim units]
    dt: float                   # Simulation timestep [s or sim units]

    # ─── Derived thermal scales ───
    omega_thermal: float        # k_B T / ℏ — thermal frequency [rad/s]
    tau_coherence: float        # ℏ / k_B T — decoherence time [s]
    A_thermal_scale: float      # √(k_B T / m) — amplitude scale (m=1) [length]

    # ─── Derived dynamics parameters ───
    bandwidth_min: float        # 1/(2·dt) — uncertainty-limited [rad/s]
    bandwidth_max: float        # ω_thermal — thermal bandwidth [rad/s]
    decay_rate: float           # Lindblad dissipation rate [1/s]
    decay_factor: float         # exp(-Γ·dt) — per-step decay

    # ─── Dimensionless ratios (for kernel use) ───
    coherence_steps: int        # τ_coherence / dt — steps to decohere

    @classmethod
    def from_temperature(
        cls,
        T: float,
        dt: float,
        c: PhysicalConstants,
        *,
        bath_coupling: float = 0.1,
        effective_mass: float = 1.0,
    ) -> "ModePhysics":
        """Compute all mode physics from temperature and timestep.

        Args:
            T: Temperature from thermodynamic domain
            dt: Simulation timestep
            c: Physical constants in simulation units
            bath_coupling: Dimensionless coupling to thermal bath.
                [CHOICE] Default 0.1 (weak coupling regime)
                [REASON] Strong coupling (≈1) thermalizes too fast.
                         Weak coupling (≈0.01) thermalizes too slow.
                         0.1 is a reasonable intermediate value.
                [NOTES] This IS a physical parameter — it describes how
                        strongly the coherence modes couple to the thermodynamic
                        heat bath. It could be measured experimentally.
            effective_mass: Mass scale for amplitude calculations.
                [CHOICE] Default 1.0 (natural units)
                [REASON] We work in units where m=1 unless specified.

        Returns:
            ModePhysics with all derived quantities.
        """
        # Handle edge cases
        if T <= 0.0:
            T = 1e-10  # Avoid division by zero; effectively T=0
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        # Thermal scales
        omega_th = (c.k_B * T) / c.hbar
        tau_coh = c.hbar / (c.k_B * T)

        # Amplitude scale: √(k_B T / m ω²) at ω = ω_thermal → √(ℏ / m ω_thermal)
        # Simplified: we use √(k_B T / m) as the characteristic amplitude
        A_scale = math.sqrt((c.k_B * T) / effective_mass) if effective_mass > 0 else 0.0

        # Bandwidth from uncertainty principle
        bw_min = 0.5 / dt
        bw_max = omega_th

        # Lindblad decay
        gamma = bath_coupling * omega_th
        decay = math.exp(-gamma * dt) if gamma * dt < 700 else 0.0

        # Coherence steps (how many dt to decohere)
        coh_steps = max(1, int(tau_coh / dt))

        return cls(
            T=T,
            dt=dt,
            omega_thermal=omega_th,
            tau_coherence=tau_coh,
            A_thermal_scale=A_scale,
            bandwidth_min=bw_min,
            bandwidth_max=bw_max,
            decay_rate=gamma,
            decay_factor=decay,
            coherence_steps=coh_steps,
        )

    def noise_floor(self, omega: float, c: PhysicalConstants, m: float = 1.0) -> float:
        """Thermal noise floor for a mode at frequency omega.

        [FORMULA] A_thermal = √((ℏ/2mω) · (2⟨n⟩ + 1))
        [REASON] A mode is only visible if its amplitude exceeds this.
        """
        return thermal_amplitude(omega, self.T, c, m)

    def is_visible(self, amplitude: float, omega: float, c: PhysicalConstants, m: float = 1.0) -> bool:
        """Check if a mode amplitude is above the thermal noise floor."""
        return amplitude > self.noise_floor(omega, c, m)

    def visibility_ratio(self, amplitude: float, omega: float, c: PhysicalConstants, m: float = 1.0) -> float:
        """Signal-to-noise ratio for mode visibility."""
        return mode_visibility_ratio(amplitude, omega, self.T, c, m)

