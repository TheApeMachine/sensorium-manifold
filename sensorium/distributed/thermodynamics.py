from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .runtime import FACES, Face, RankConfig, TickPhase, Transport
from .triton_kernels import advance_interior_halo, jacobi_step_halo, pack_halo_face

try:
    from .metal_kernels import (
        advance_interior_halo_metal,
        jacobi_step_halo_metal,
        metal_distributed_available,
        pack_halo_face_scalar_metal,
    )
except Exception:

    def metal_distributed_available() -> bool:
        return False

    def pack_halo_face_scalar_metal(
        field: torch.Tensor, *, face: Face, halo: int
    ) -> torch.Tensor:
        del face, halo
        return field

    def jacobi_step_halo_metal(
        phi: torch.Tensor,
        rhs: torch.Tensor,
        *,
        halo_xm: torch.Tensor,
        halo_xp: torch.Tensor,
        halo_ym: torch.Tensor,
        halo_yp: torch.Tensor,
        halo_zm: torch.Tensor,
        halo_zp: torch.Tensor,
        dx: float,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del halo_xm, halo_xp, halo_ym, halo_yp, halo_zm, halo_zp, dx
        out_t = out if out is not None else phi.clone()
        out_t.copy_(rhs)
        return out_t

    def advance_interior_halo_metal(
        *,
        rho_ext: torch.Tensor,
        mom_ext: torch.Tensor,
        e_ext: torch.Tensor,
        phi_ext: torch.Tensor,
        dt: float,
        dx: float,
        gamma: float,
        rho_min: float,
        viscosity: float,
        thermal_diffusivity: float,
        halo: int,
        out_rho: torch.Tensor,
        out_mom: torch.Tensor,
        out_e: torch.Tensor,
    ) -> None:
        del rho_ext, mom_ext, e_ext, phi_ext, dt, dx, gamma, rho_min, viscosity
        del thermal_diffusivity, halo
        out_rho.zero_()
        out_mom.zero_()
        out_e.zero_()


@dataclass(frozen=True)
class DistributedThermoConfig:
    grid_spacing: float
    dt_max: float = 0.01
    gamma: float = 1.4
    rho_min: float = 1e-5
    c_v: float = 1.0
    viscosity: float = 2e-3
    thermal_diffusivity: float = 2e-3
    gravity_g: float = 1.0
    gravity_jacobi_iters: int = 64


class DistributedThermodynamicsDomain:
    def __init__(
        self,
        *,
        config: DistributedThermoConfig,
        rank_config: RankConfig,
        transport: Transport,
        device: torch.device,
    ) -> None:
        self.config = config
        self.rank_config = rank_config
        self.transport = transport
        self.device = device
        self.dtype = torch.float32
        gx, gy, gz = rank_config.local_grid_size
        self.rho_field = torch.full(
            (gx, gy, gz),
            fill_value=config.rho_min,
            device=device,
            dtype=self.dtype,
        )
        self.mom_field = torch.zeros((gx, gy, gz, 3), device=device, dtype=self.dtype)
        self.e_int_field = torch.zeros((gx, gy, gz), device=device, dtype=self.dtype)
        self.gravity_potential = torch.zeros(
            (gx, gy, gz), device=device, dtype=self.dtype
        )
        self._halo_scalar: dict[str, dict[Face, torch.Tensor]] = {
            "rho": self._alloc_scalar_halos(),
            "e_int": self._alloc_scalar_halos(),
            "phi": self._alloc_scalar_halos(),
        }
        self._halo_vec3: dict[str, dict[Face, torch.Tensor]] = {
            "mom": self._alloc_vec_halos(3),
        }

    def _alloc_scalar_halos(self) -> dict[Face, torch.Tensor]:
        h = self.rank_config.halo_thickness
        gx, gy, gz = self.rank_config.local_grid_size
        return {
            "x-": torch.zeros((h, gy, gz), device=self.device, dtype=self.dtype),
            "x+": torch.zeros((h, gy, gz), device=self.device, dtype=self.dtype),
            "y-": torch.zeros((gx, h, gz), device=self.device, dtype=self.dtype),
            "y+": torch.zeros((gx, h, gz), device=self.device, dtype=self.dtype),
            "z-": torch.zeros((gx, gy, h), device=self.device, dtype=self.dtype),
            "z+": torch.zeros((gx, gy, h), device=self.device, dtype=self.dtype),
        }

    def _alloc_vec_halos(self, channels: int) -> dict[Face, torch.Tensor]:
        h = self.rank_config.halo_thickness
        gx, gy, gz = self.rank_config.local_grid_size
        return {
            "x-": torch.zeros(
                (h, gy, gz, channels), device=self.device, dtype=self.dtype
            ),
            "x+": torch.zeros(
                (h, gy, gz, channels), device=self.device, dtype=self.dtype
            ),
            "y-": torch.zeros(
                (gx, h, gz, channels), device=self.device, dtype=self.dtype
            ),
            "y+": torch.zeros(
                (gx, h, gz, channels), device=self.device, dtype=self.dtype
            ),
            "z-": torch.zeros(
                (gx, gy, h, channels), device=self.device, dtype=self.dtype
            ),
            "z+": torch.zeros(
                (gx, gy, h, channels), device=self.device, dtype=self.dtype
            ),
        }

    def _pack_faces(self, tensor: torch.Tensor) -> dict[Face, torch.Tensor]:
        h = self.rank_config.halo_thickness
        if self.device.type == "mps" and metal_distributed_available():
            if tensor.ndim == 3:
                return {
                    face: pack_halo_face_scalar_metal(tensor, face=face, halo=h)
                    for face in FACES
                }
            if tensor.ndim == 4 and int(tensor.shape[-1]) == 3:
                faces: dict[Face, torch.Tensor] = {}
                for face in FACES:
                    chans = [
                        pack_halo_face_scalar_metal(tensor[..., c], face=face, halo=h)
                        for c in range(3)
                    ]
                    faces[face] = torch.stack(chans, dim=-1)
                return faces
        if self.device.type == "cuda":
            return {face: pack_halo_face(tensor, face=face, halo=h) for face in FACES}
        return {
            "x-": tensor[:h, ...].contiguous(),
            "x+": tensor[-h:, ...].contiguous(),
            "y-": tensor[:, :h, ...].contiguous(),
            "y+": tensor[:, -h:, ...].contiguous(),
            "z-": tensor[:, :, :h, ...].contiguous(),
            "z+": tensor[:, :, -h:, ...].contiguous(),
        }

    def _exchange_scalar_halos(
        self, name: str, tensor: torch.Tensor, phase: TickPhase, tick: int
    ) -> None:
        send = self._pack_faces(tensor)
        recv = self.transport.exchange_halos(
            tick=tick,
            phase=phase,
            send_buffers=send,
            neighbors=self.rank_config.neighbor_ids,
            device=self.device,
        )
        halos = self._halo_scalar[name]
        for face in FACES:
            if face in recv:
                halos[face].copy_(recv[face])
            else:
                halos[face].copy_(send[face])

    def _exchange_vec_halos(
        self, name: str, tensor: torch.Tensor, phase: TickPhase, tick: int
    ) -> None:
        send = self._pack_faces(tensor)
        recv = self.transport.exchange_halos(
            tick=tick,
            phase=phase,
            send_buffers=send,
            neighbors=self.rank_config.neighbor_ids,
            device=self.device,
        )
        halos = self._halo_vec3[name]
        for face in FACES:
            if face in recv:
                halos[face].copy_(recv[face])
            else:
                halos[face].copy_(send[face])

    def exchange_grid_halos(self, tick: int) -> None:
        self._exchange_scalar_halos("rho", self.rho_field, "grid_halo", tick)
        self._exchange_vec_halos("mom", self.mom_field, "grid_halo", tick)
        self._exchange_scalar_halos("e_int", self.e_int_field, "grid_halo", tick)

    def _extended_scalar(
        self, interior: torch.Tensor, halos: dict[Face, torch.Tensor]
    ) -> torch.Tensor:
        h = self.rank_config.halo_thickness
        gx, gy, gz = self.rank_config.local_grid_size
        ext = torch.empty(
            (gx + 2 * h, gy + 2 * h, gz + 2 * h), device=self.device, dtype=self.dtype
        )
        ext[h : h + gx, h : h + gy, h : h + gz] = interior
        ext[:h, h : h + gy, h : h + gz] = halos["x-"]
        ext[h + gx :, h : h + gy, h : h + gz] = halos["x+"]
        ext[h : h + gx, :h, h : h + gz] = halos["y-"]
        ext[h : h + gx, h + gy :, h : h + gz] = halos["y+"]
        ext[h : h + gx, h : h + gy, :h] = halos["z-"]
        ext[h : h + gx, h : h + gy, h + gz :] = halos["z+"]
        ext[:h, :h, :] = ext[h : h + 1, :h, :]
        ext[:h, h + gy :, :] = ext[h : h + 1, h + gy :, :]
        ext[h + gx :, :h, :] = ext[h + gx - 1 : h + gx, :h, :]
        ext[h + gx :, h + gy :, :] = ext[h + gx - 1 : h + gx, h + gy :, :]
        ext[:, :, :h] = ext[:, :, h : h + 1]
        ext[:, :, h + gz :] = ext[:, :, h + gz - 1 : h + gz]
        return ext

    def _extended_vec3(
        self, interior: torch.Tensor, halos: dict[Face, torch.Tensor]
    ) -> torch.Tensor:
        h = self.rank_config.halo_thickness
        gx, gy, gz = self.rank_config.local_grid_size
        ext = torch.empty(
            (gx + 2 * h, gy + 2 * h, gz + 2 * h, 3),
            device=self.device,
            dtype=self.dtype,
        )
        ext[h : h + gx, h : h + gy, h : h + gz, :] = interior
        ext[:h, h : h + gy, h : h + gz, :] = halos["x-"]
        ext[h + gx :, h : h + gy, h : h + gz, :] = halos["x+"]
        ext[h : h + gx, :h, h : h + gz, :] = halos["y-"]
        ext[h : h + gx, h + gy :, h : h + gz, :] = halos["y+"]
        ext[h : h + gx, h : h + gy, :h, :] = halos["z-"]
        ext[h : h + gx, h : h + gy, h + gz :, :] = halos["z+"]
        ext[:h, :h, :, :] = ext[h : h + 1, :h, :, :]
        ext[:h, h + gy :, :, :] = ext[h : h + 1, h + gy :, :, :]
        ext[h + gx :, :h, :, :] = ext[h + gx - 1 : h + gx, :h, :, :]
        ext[h + gx :, h + gy :, :, :] = ext[h + gx - 1 : h + gx, h + gy :, :, :]
        ext[:, :, :h, :] = ext[:, :, h : h + 1, :]
        ext[:, :, h + gz :, :] = ext[:, :, h + gz - 1 : h + gz, :]
        return ext

    def _laplacian(self, ext: torch.Tensor) -> torch.Tensor:
        h = self.rank_config.halo_thickness
        c = ext[h:-h, h:-h, h:-h]
        xp = ext[h + 1 :, h:-h, h:-h]
        xm = ext[: -h - 1, h:-h, h:-h]
        yp = ext[h:-h, h + 1 :, h:-h]
        ym = ext[h:-h, : -h - 1, h:-h]
        zp = ext[h:-h, h:-h, h + 1 :]
        zm = ext[h:-h, h:-h, : -h - 1]
        inv_dx2 = 1.0 / (self.config.grid_spacing * self.config.grid_spacing)
        return (xp + xm + yp + ym + zp + zm - 6.0 * c) * inv_dx2

    def solve_gravity(self, tick: int) -> None:
        phi = self.gravity_potential
        rhs = (
            4.0
            * math.pi
            * self.config.gravity_g
            * (self.rho_field - self.rho_field.mean())
        )
        h2 = self.config.grid_spacing * self.config.grid_spacing
        for it in range(self.config.gravity_jacobi_iters):
            self._exchange_scalar_halos("phi", phi, "gravity_halo", tick * 10000 + it)
            halos = self._halo_scalar["phi"]
            if self.device.type == "cuda":
                phi = jacobi_step_halo(
                    phi,
                    rhs,
                    halo_xm=halos["x-"],
                    halo_xp=halos["x+"],
                    halo_ym=halos["y-"],
                    halo_yp=halos["y+"],
                    halo_zm=halos["z-"],
                    halo_zp=halos["z+"],
                    dx=self.config.grid_spacing,
                )
            elif self.device.type == "mps" and metal_distributed_available():
                phi = jacobi_step_halo_metal(
                    phi,
                    rhs,
                    halo_xm=halos["x-"],
                    halo_xp=halos["x+"],
                    halo_ym=halos["y-"],
                    halo_yp=halos["y+"],
                    halo_zm=halos["z-"],
                    halo_zp=halos["z+"],
                    dx=self.config.grid_spacing,
                )
            else:
                ext = self._extended_scalar(phi, halos)
                phi = (
                    ext[2:, 1:-1, 1:-1]
                    + ext[:-2, 1:-1, 1:-1]
                    + ext[1:-1, 2:, 1:-1]
                    + ext[1:-1, :-2, 1:-1]
                    + ext[1:-1, 1:-1, 2:]
                    + ext[1:-1, 1:-1, :-2]
                    - h2 * rhs
                ) / 6.0
        self.gravity_potential.copy_(phi)

    def advance_grid_interior(self, dt: float) -> None:
        dt_eff = min(float(dt), self.config.dt_max)
        h = self.rank_config.halo_thickness
        rho_ext = self._extended_scalar(self.rho_field, self._halo_scalar["rho"])
        mom_ext = self._extended_vec3(self.mom_field, self._halo_vec3["mom"])
        e_ext = self._extended_scalar(self.e_int_field, self._halo_scalar["e_int"])
        phi_ext = self._extended_scalar(
            self.gravity_potential, self._halo_scalar["phi"]
        )
        new_rho = torch.empty_like(self.rho_field)
        new_mom = torch.empty_like(self.mom_field)
        new_e = torch.empty_like(self.e_int_field)

        if self.device.type == "mps" and metal_distributed_available():
            advance_interior_halo_metal(
                rho_ext=rho_ext,
                mom_ext=mom_ext,
                e_ext=e_ext,
                phi_ext=phi_ext,
                dt=dt_eff,
                dx=self.config.grid_spacing,
                gamma=self.config.gamma,
                rho_min=self.config.rho_min,
                viscosity=self.config.viscosity,
                thermal_diffusivity=self.config.thermal_diffusivity,
                halo=h,
                out_rho=new_rho,
                out_mom=new_mom,
                out_e=new_e,
            )
        else:
            advance_interior_halo(
                rho_ext=rho_ext,
                mom_ext=mom_ext,
                e_ext=e_ext,
                phi_ext=phi_ext,
                dt=dt_eff,
                dx=self.config.grid_spacing,
                gamma=self.config.gamma,
                rho_min=self.config.rho_min,
                viscosity=self.config.viscosity,
                thermal_diffusivity=self.config.thermal_diffusivity,
                halo=h,
                out_rho=new_rho,
                out_mom=new_mom,
                out_e=new_e,
            )

        self.rho_field.copy_(new_rho)
        self.mom_field.copy_(new_mom)
        self.e_int_field.copy_(new_e)

    def sample_grid(
        self, positions_local: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gx, gy, gz = self.rank_config.local_grid_size
        dx = self.config.grid_spacing
        ix = torch.clamp(
            (positions_local[:, 0] / dx).floor().to(torch.int64), 0, gx - 1
        )
        iy = torch.clamp(
            (positions_local[:, 1] / dx).floor().to(torch.int64), 0, gy - 1
        )
        iz = torch.clamp(
            (positions_local[:, 2] / dx).floor().to(torch.int64), 0, gz - 1
        )
        vel = self.mom_field[ix, iy, iz, :] / self.rho_field[ix, iy, iz].clamp_min(
            self.config.rho_min
        ).unsqueeze(-1)
        heat = self.e_int_field[ix, iy, iz] / self.rho_field[ix, iy, iz].clamp_min(
            self.config.rho_min
        )
        return vel, heat
