from __future__ import annotations

from dataclasses import dataclass

import torch

from .runtime import Face, RankConfig
from .triton_kernels import classify_migration_faces

try:
    from .metal_kernels import (
        classify_migration_faces_metal,
        metal_distributed_available,
    )
except Exception:

    def metal_distributed_available() -> bool:
        return False

    def classify_migration_faces_metal(
        positions: torch.Tensor,
        *,
        lo: tuple[float, float, float],
        hi: tuple[float, float, float],
    ) -> torch.Tensor:
        del lo, hi
        return torch.zeros(
            (positions.shape[0],), device=positions.device, dtype=torch.int32
        )


PARTICLE_FIELDS: tuple[str, ...] = (
    "positions",
    "velocities",
    "masses",
    "heats",
    "energies",
    "excitations",
    "phase",
)


@dataclass
class ParticleBatch:
    positions: torch.Tensor
    velocities: torch.Tensor
    masses: torch.Tensor
    heats: torch.Tensor
    energies: torch.Tensor
    excitations: torch.Tensor
    phase: torch.Tensor

    @classmethod
    def from_state(
        cls, state: dict[str, torch.Tensor], device: torch.device
    ) -> "ParticleBatch":
        return cls(
            positions=state["positions"]
            .to(device=device, dtype=torch.float32)
            .contiguous(),
            velocities=state["velocities"]
            .to(device=device, dtype=torch.float32)
            .contiguous(),
            masses=state["masses"].to(device=device, dtype=torch.float32).contiguous(),
            heats=state["heats"].to(device=device, dtype=torch.float32).contiguous(),
            energies=state.get("energies", state["energy_osc"])
            .to(
                device=device,
                dtype=torch.float32,
            )
            .contiguous(),
            excitations=state.get("excitations", state["omega"])
            .to(
                device=device,
                dtype=torch.float32,
            )
            .contiguous(),
            phase=state.get("phase", torch.zeros_like(state["masses"]))
            .to(
                device=device,
                dtype=torch.float32,
            )
            .contiguous(),
        )

    def to_state(self) -> dict[str, torch.Tensor]:
        return {
            "positions": self.positions,
            "velocities": self.velocities,
            "masses": self.masses,
            "heats": self.heats,
            "energies": self.energies,
            "excitations": self.excitations,
            "phase": self.phase,
        }

    def size(self) -> int:
        return int(self.positions.shape[0])


class ParticleMigrator:
    def __init__(
        self,
        *,
        rank_config: RankConfig,
        grid_spacing: float,
        device: torch.device,
    ) -> None:
        self.rank_config = rank_config
        self.grid_spacing = float(grid_spacing)
        self.device = device

    def advance_particles(
        self,
        batch: ParticleBatch,
        *,
        dt: float,
        grid_velocity: torch.Tensor,
        grid_heat: torch.Tensor,
    ) -> ParticleBatch:
        batch.velocities = grid_velocity.to(self.device)
        batch.positions = batch.positions + dt * batch.velocities
        batch.heats = 0.98 * batch.heats + 0.02 * grid_heat.to(self.device)
        return batch

    def split_outbound(
        self,
        batch: ParticleBatch,
    ) -> tuple[ParticleBatch, dict[Face, dict[str, torch.Tensor]]]:
        origin = torch.tensor(
            self.rank_config.tile_origin, device=self.device, dtype=torch.float32
        )
        local = torch.tensor(
            self.rank_config.local_grid_size, device=self.device, dtype=torch.float32
        )
        lo = origin * self.grid_spacing
        hi = (origin + local) * self.grid_spacing

        pos = batch.positions
        if self.device.type == "mps" and metal_distributed_available():
            codes = classify_migration_faces_metal(
                pos,
                lo=(float(lo[0]), float(lo[1]), float(lo[2])),
                hi=(float(hi[0]), float(hi[1]), float(hi[2])),
            )
        else:
            codes = classify_migration_faces(
                pos,
                lo=(float(lo[0]), float(lo[1]), float(lo[2])),
                hi=(float(hi[0]), float(hi[1]), float(hi[2])),
            )
        masks: dict[Face, torch.Tensor] = {
            "x-": codes == 1,
            "x+": codes == 2,
            "y-": codes == 3,
            "y+": codes == 4,
            "z-": codes == 5,
            "z+": codes == 6,
        }
        outbound_any = torch.zeros(
            (batch.size(),), device=self.device, dtype=torch.bool
        )
        for m in masks.values():
            outbound_any = outbound_any | m

        kept = self._select(batch, ~outbound_any)
        payloads: dict[Face, dict[str, torch.Tensor]] = {}
        for face, mask in masks.items():
            payloads[face] = self._payload_dict(self._select(batch, mask))
        return kept, payloads

    def merge_inbound(
        self,
        batch: ParticleBatch,
        inbound_payloads: dict[Face, dict[str, torch.Tensor]],
    ) -> ParticleBatch:
        chunks: list[ParticleBatch] = [batch]
        for payload in inbound_payloads.values():
            if not payload:
                continue
            if payload["positions"].numel() == 0:
                continue
            chunks.append(
                ParticleBatch(
                    positions=payload["positions"],
                    velocities=payload["velocities"],
                    masses=payload["masses"],
                    heats=payload["heats"],
                    energies=payload["energies"],
                    excitations=payload["excitations"],
                    phase=payload["phase"],
                )
            )
        if len(chunks) == 1:
            return batch
        return ParticleBatch(
            positions=torch.cat([c.positions for c in chunks], dim=0),
            velocities=torch.cat([c.velocities for c in chunks], dim=0),
            masses=torch.cat([c.masses for c in chunks], dim=0),
            heats=torch.cat([c.heats for c in chunks], dim=0),
            energies=torch.cat([c.energies for c in chunks], dim=0),
            excitations=torch.cat([c.excitations for c in chunks], dim=0),
            phase=torch.cat([c.phase for c in chunks], dim=0),
        )

    def _select(self, batch: ParticleBatch, mask: torch.Tensor) -> ParticleBatch:
        return ParticleBatch(
            positions=batch.positions[mask],
            velocities=batch.velocities[mask],
            masses=batch.masses[mask],
            heats=batch.heats[mask],
            energies=batch.energies[mask],
            excitations=batch.excitations[mask],
            phase=batch.phase[mask],
        )

    def _payload_dict(self, batch: ParticleBatch) -> dict[str, torch.Tensor]:
        return {
            "positions": batch.positions,
            "velocities": batch.velocities,
            "masses": batch.masses,
            "heats": batch.heats,
            "energies": batch.energies,
            "excitations": batch.excitations,
            "phase": batch.phase,
        }
