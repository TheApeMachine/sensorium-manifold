from dataclasses import dataclass
import torch


@dataclass
class CarrierState:
    """Spectral carriers that mediate entanglement in frequency space.
    
    Carriers exist purely in frequency space - they have NO spatial position.
    They are characterized by:
    - frequency (ω): the center frequency they couple to
    - gate_width (σ): tolerance/specialization (how broad/narrow their coupling)
    - amplitude: complex magnitude of accumulated drive
    - phase: complex phase of the carrier
    
    Coupling to oscillators is determined by frequency alignment:
        tuning = exp(-(ω_osc - ω_carrier)² / σ²)
    
    For visualization purposes, carrier positions are computed by the
    dashboard based on the centroid of coupled oscillators.
    """
    frequencies: torch.Tensor   # (M,) - carrier center frequencies
    gate_widths: torch.Tensor   # (M,) - coupling tolerance (σ)
    amplitudes: torch.Tensor    # (M,) - complex amplitude magnitude
    phases: torch.Tensor        # (M,) - complex amplitude phase
    
    @classmethod
    def empty(cls, device: str, dtype: torch.dtype) -> "CarrierState":
        return cls(
            frequencies=torch.empty(0, device=device, dtype=dtype),
            gate_widths=torch.empty(0, device=device, dtype=dtype),
            amplitudes=torch.empty(0, device=device, dtype=dtype),
            phases=torch.empty(0, device=device, dtype=dtype),
        )
    
    @property
    def num_carriers(self) -> int:
        return self.frequencies.shape[0]
