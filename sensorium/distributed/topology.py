from __future__ import annotations

from dataclasses import dataclass

from .runtime import Face, RankConfig


@dataclass(frozen=True)
class CartesianTopology:
    global_grid_size: tuple[int, int, int]
    tile_grid_shape: tuple[int, int, int]
    halo_thickness: int = 1
    periodic: bool = True

    @property
    def world_size(self) -> int:
        tx, ty, tz = self.tile_grid_shape
        return int(tx * ty * tz)

    def rank_to_coords(self, rank_id: int) -> tuple[int, int, int]:
        tx, ty, tz = self.tile_grid_shape
        if rank_id < 0 or rank_id >= self.world_size:
            raise ValueError(f"rank_id out of range: {rank_id}")
        x = rank_id // (ty * tz)
        rem = rank_id % (ty * tz)
        y = rem // tz
        z = rem % tz
        return x, y, z

    def coords_to_rank(self, coords: tuple[int, int, int]) -> int:
        tx, ty, tz = self.tile_grid_shape
        x, y, z = coords
        if self.periodic:
            x %= tx
            y %= ty
            z %= tz
        if not (0 <= x < tx and 0 <= y < ty and 0 <= z < tz):
            return -1
        return int(x * ty * tz + y * tz + z)

    def build_rank_config(self, rank_id: int) -> RankConfig:
        rx, ry, rz = self.rank_to_coords(rank_id)
        gx, gy, gz = self.global_grid_size
        tx, ty, tz = self.tile_grid_shape
        if gx % tx != 0 or gy % ty != 0 or gz % tz != 0:
            raise ValueError(
                "global_grid_size must be divisible by tile_grid_shape: "
                f"global={self.global_grid_size} tiles={self.tile_grid_shape}"
            )
        local = (gx // tx, gy // ty, gz // tz)
        origin = (rx * local[0], ry * local[1], rz * local[2])
        neighbors: dict[Face, int] = {
            "x-": self.coords_to_rank((rx - 1, ry, rz)),
            "x+": self.coords_to_rank((rx + 1, ry, rz)),
            "y-": self.coords_to_rank((rx, ry - 1, rz)),
            "y+": self.coords_to_rank((rx, ry + 1, rz)),
            "z-": self.coords_to_rank((rx, ry, rz - 1)),
            "z+": self.coords_to_rank((rx, ry, rz + 1)),
        }
        return RankConfig(
            rank_id=rank_id,
            neighbor_ids=neighbors,
            tile_origin=origin,
            local_grid_size=local,
            halo_thickness=self.halo_thickness,
        )
