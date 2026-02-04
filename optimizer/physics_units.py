"""Unit systems + physical constants (auditable, non-tuneable).

This module defines:
- CODATA physical constants in SI units (universal constants).
- An explicit base-unit mapping from simulation units -> SI units.
- Deterministic conversion of physical constants into simulation units.

------------------------------------------------------------------------------
COMMENT CONVENTION (physics choices)
------------------------------------------------------------------------------
  # [CHOICE] <name>
  # [FORMULA] <math / equation / mapping>
  # [REASON] <brief why this form/value>
  # [NOTES] <brief caveats, assumptions, invariants, TODOs>
------------------------------------------------------------------------------
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
    # [NOTES] if you want uncertainty propagation, we can add ± here later.
    G: float = 6.67430e-11

    # [CHOICE] Boltzmann constant (SI)
    # [FORMULA] k_B = 1.380649e-23 J K^-1 (exact)
    # [REASON] universal constant (defined)
    k_B: float = 1.380649e-23

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


def assert_finite_constants(c: PhysicalConstants) -> None:
    """Fail loudly if constants are non-finite."""
    for name, v in (("G", c.G), ("k_B", c.k_B), ("sigma_SB", c.sigma_SB), ("hbar", c.hbar)):
        if not math.isfinite(float(v)):
            raise ValueError(f"Non-finite constant {name}={v!r} (source={c.source})")

