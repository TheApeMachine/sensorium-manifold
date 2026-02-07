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
        return self._split_by_codes(batch, codes)

    def _split_by_codes(
        self, batch: ParticleBatch, codes: torch.Tensor
    ) -> tuple[ParticleBatch, dict[Face, dict[str, torch.Tensor]]]:
        if codes.numel() == 0:
            empty = self._select(
                batch, torch.zeros((0,), device=self.device, dtype=torch.bool)
            )
            return empty, {
                face: self._payload_dict(empty)
                for face in ("x-", "x+", "y-", "y+", "z-", "z+")
            }

        order = torch.argsort(codes.to(torch.int64), stable=True)
        sorted_codes = codes[order]
        sorted_batch = ParticleBatch(
            positions=batch.positions[order],
            velocities=batch.velocities[order],
            masses=batch.masses[order],
            heats=batch.heats[order],
            energies=batch.energies[order],
            excitations=batch.excitations[order],
            phase=batch.phase[order],
        )

        counts = torch.bincount(sorted_codes.to(torch.int64), minlength=7)
        offsets = torch.cumsum(counts, dim=0)

        def sl(code: int) -> slice:
            end = int(offsets[code].item())
            start = int(offsets[code - 1].item()) if code > 0 else 0
            return slice(start, end)

        kept = ParticleBatch(
            positions=sorted_batch.positions[sl(0)],
            velocities=sorted_batch.velocities[sl(0)],
            masses=sorted_batch.masses[sl(0)],
            heats=sorted_batch.heats[sl(0)],
            energies=sorted_batch.energies[sl(0)],
            excitations=sorted_batch.excitations[sl(0)],
            phase=sorted_batch.phase[sl(0)],
        )
        code_to_face: dict[int, Face] = {
            1: "x-",
            2: "x+",
            3: "y-",
            4: "y+",
            5: "z-",
            6: "z+",
        }
        payloads: dict[Face, dict[str, torch.Tensor]] = {}
        for code, face in code_to_face.items():
            s = sl(code)
            payloads[face] = self._payload_dict(
                ParticleBatch(
                    positions=sorted_batch.positions[s],
                    velocities=sorted_batch.velocities[s],
                    masses=sorted_batch.masses[s],
                    heats=sorted_batch.heats[s],
                    energies=sorted_batch.energies[s],
                    excitations=sorted_batch.excitations[s],
                    phase=sorted_batch.phase[s],
                )
            )
        return kept, payloads

    def merge_inbound(
        self,
        batch: ParticleBatch,
        inbound_payloads: dict[Face, dict[str, torch.Tensor]],
    ) -> ParticleBatch:
        local_n = batch.size()
        inbound_faces: tuple[Face, ...] = ("x-", "x+", "y-", "y+", "z-", "z+")
        inbound_counts: dict[Face, int] = {}
        total = local_n
        for face in inbound_faces:
            payload = inbound_payloads.get(face, {})
            n = int(
                payload.get("positions", torch.empty(0, device=self.device)).shape[0]
            )
            inbound_counts[face] = n
            total += n

        if total == local_n:
            return batch

        positions = batch.positions.new_empty((total, 3))
        velocities = batch.velocities.new_empty((total, 3))
        masses = batch.masses.new_empty((total,))
        heats = batch.heats.new_empty((total,))
        energies = batch.energies.new_empty((total,))
        excitations = batch.excitations.new_empty((total,))
        phase = batch.phase.new_empty((total,))

        write = 0
        if local_n > 0:
            end = write + local_n
            positions[write:end].copy_(batch.positions)
            velocities[write:end].copy_(batch.velocities)
            masses[write:end].copy_(batch.masses)
            heats[write:end].copy_(batch.heats)
            energies[write:end].copy_(batch.energies)
            excitations[write:end].copy_(batch.excitations)
            phase[write:end].copy_(batch.phase)
            write = end

        for face in inbound_faces:
            n = inbound_counts[face]
            if n <= 0:
                continue
            payload = inbound_payloads[face]
            end = write + n
            positions[write:end].copy_(payload["positions"])
            velocities[write:end].copy_(payload["velocities"])
            masses[write:end].copy_(payload["masses"])
            heats[write:end].copy_(payload["heats"])
            energies[write:end].copy_(payload["energies"])
            excitations[write:end].copy_(payload["excitations"])
            phase[write:end].copy_(payload["phase"])
            write = end

        return ParticleBatch(
            positions=positions,
            velocities=velocities,
            masses=masses,
            heats=heats,
            energies=energies,
            excitations=excitations,
            phase=phase,
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
