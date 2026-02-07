from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist

Face = Literal["x-", "x+", "y-", "y+", "z-", "z+"]
TickPhase = Literal[
    "grid_halo",
    "gravity_halo",
    "particle_migration",
    "mode_route",
]

FACES: tuple[Face, ...] = ("x-", "x+", "y-", "y+", "z-", "z+")
OPPOSITE_FACE: dict[Face, Face] = {
    "x-": "x+",
    "x+": "x-",
    "y-": "y+",
    "y+": "y-",
    "z-": "z+",
    "z+": "z-",
}

_PHASE_ID: dict[TickPhase, int] = {
    "grid_halo": 1,
    "gravity_halo": 2,
    "particle_migration": 3,
    "mode_route": 4,
}
_FACE_ID: dict[Face, int] = {face: idx for idx, face in enumerate(FACES)}


@dataclass(frozen=True)
class RankConfig:
    rank_id: int
    neighbor_ids: dict[Face, int]
    tile_origin: tuple[int, int, int]
    local_grid_size: tuple[int, int, int]
    halo_thickness: int


@dataclass(frozen=True)
class TickEnvelope:
    tick: int
    phase: TickPhase
    src_rank: int
    dst_rank: int


class Transport(ABC):
    @abstractmethod
    def exchange_halos(
        self,
        *,
        tick: int,
        phase: TickPhase,
        send_buffers: dict[Face, torch.Tensor],
        neighbors: dict[Face, int],
        device: torch.device,
    ) -> dict[Face, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def exchange_particle_payloads(
        self,
        *,
        tick: int,
        payloads: dict[Face, dict[str, torch.Tensor]],
        neighbors: dict[Face, int],
        device: torch.device,
    ) -> dict[Face, dict[str, torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def route_mode_payloads(
        self,
        *,
        tick: int,
        payloads_by_rank: dict[int, dict[str, torch.Tensor]],
        device: torch.device,
    ) -> list[dict[str, torch.Tensor]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def rank(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def world_size(self) -> int:
        raise NotImplementedError


class LoopbackTransport(Transport):
    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    def exchange_halos(
        self,
        *,
        tick: int,
        phase: TickPhase,
        send_buffers: dict[Face, torch.Tensor],
        neighbors: dict[Face, int],
        device: torch.device,
    ) -> dict[Face, torch.Tensor]:
        del tick, phase, neighbors, device
        return {face: tensor.clone() for face, tensor in send_buffers.items()}

    def exchange_particle_payloads(
        self,
        *,
        tick: int,
        payloads: dict[Face, dict[str, torch.Tensor]],
        neighbors: dict[Face, int],
        device: torch.device,
    ) -> dict[Face, dict[str, torch.Tensor]]:
        del tick, neighbors, device
        return {
            face: {key: value.clone() for key, value in payload.items()}
            for face, payload in payloads.items()
        }

    def route_mode_payloads(
        self,
        *,
        tick: int,
        payloads_by_rank: dict[int, dict[str, torch.Tensor]],
        device: torch.device,
    ) -> list[dict[str, torch.Tensor]]:
        del tick, device
        own = payloads_by_rank.get(0, {})
        if not own:
            return []
        return [{key: value.clone() for key, value in own.items()}]


class TorchDistributedTransport(Transport):
    def __init__(
        self,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        self._group = process_group
        self._backend = dist.get_backend(process_group)
        self._rank = dist.get_rank(process_group)
        self._world_size = dist.get_world_size(process_group)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def exchange_halos(
        self,
        *,
        tick: int,
        phase: TickPhase,
        send_buffers: dict[Face, torch.Tensor],
        neighbors: dict[Face, int],
        device: torch.device,
    ) -> dict[Face, torch.Tensor]:
        recv_buffers: dict[Face, torch.Tensor] = {}
        requests: list[dist.Work] = []
        staged: dict[Face, torch.Tensor] = {}
        for face in FACES:
            neighbor = neighbors.get(face, -1)
            if neighbor < 0 or face not in send_buffers:
                continue
            send_tensor = send_buffers[face].contiguous()
            if self._use_direct_device_transfer(send_tensor):
                send_buf = send_tensor
                recv_buf = torch.empty_like(send_tensor)
            else:
                send_buf = send_tensor.to("cpu", non_blocking=False)
                recv_buf = torch.empty_like(send_buf)
                staged[face] = recv_buf
            tag_out = _message_tag(tick=tick, phase=phase, face=face)
            tag_in = _message_tag(tick=tick, phase=phase, face=OPPOSITE_FACE[face])
            send_req = dist.isend(
                tensor=send_buf,
                dst=neighbor,
                tag=tag_out,
                group=self._group,
            )
            recv_req = dist.irecv(
                tensor=recv_buf,
                src=neighbor,
                tag=tag_in,
                group=self._group,
            )
            if send_req is not None:
                requests.append(send_req)
            if recv_req is not None:
                requests.append(recv_req)
            recv_buffers[face] = recv_buf
        for req in requests:
            req.wait()
        for face, recv in recv_buffers.items():
            if face in staged:
                recv_buffers[face] = recv.to(device, non_blocking=False)
        return recv_buffers

    def exchange_particle_payloads(
        self,
        *,
        tick: int,
        payloads: dict[Face, dict[str, torch.Tensor]],
        neighbors: dict[Face, int],
        device: torch.device,
    ) -> dict[Face, dict[str, torch.Tensor]]:
        del tick
        received: dict[Face, dict[str, torch.Tensor]] = {}
        for face in FACES:
            neighbor = neighbors.get(face, -1)
            if neighbor < 0:
                continue
            send_obj = _cpu_payload(payloads.get(face, {}))
            recv_holder: list[dict[str, torch.Tensor]] = [{}]
            if self.rank < neighbor:
                dist.send_object_list([send_obj], dst=neighbor, group=self._group)
                dist.recv_object_list(recv_holder, src=neighbor, group=self._group)
            else:
                dist.recv_object_list(recv_holder, src=neighbor, group=self._group)
                dist.send_object_list([send_obj], dst=neighbor, group=self._group)
            received[face] = {
                key: value.to(device, non_blocking=False)
                for key, value in recv_holder[0].items()
            }
        return received

    def route_mode_payloads(
        self,
        *,
        tick: int,
        payloads_by_rank: dict[int, dict[str, torch.Tensor]],
        device: torch.device,
    ) -> list[dict[str, torch.Tensor]]:
        del tick
        outbound = {
            dst: _cpu_payload(payload)
            for dst, payload in payloads_by_rank.items()
            if payload
        }
        gathered: list[dict[int, dict[str, torch.Tensor]] | None] = [
            None for _ in range(self.world_size)
        ]
        dist.all_gather_object(gathered, outbound, group=self._group)
        inbound: list[dict[str, torch.Tensor]] = []
        for src_map in gathered:
            if not src_map:
                continue
            payload = src_map.get(self.rank)
            if not payload:
                continue
            inbound.append(
                {
                    key: value.to(device, non_blocking=False)
                    for key, value in payload.items()
                }
            )
        return inbound

    def _use_direct_device_transfer(self, tensor: torch.Tensor) -> bool:
        backend = str(self._backend).lower()
        return tensor.device.type == "cuda" and backend == "nccl"


@dataclass
class DistributedCoordinator:
    phase_order: tuple[TickPhase, ...] = (
        "grid_halo",
        "gravity_halo",
        "particle_migration",
        "mode_route",
    )

    def envelopes_for_tick(self, tick: int, world_size: int) -> list[TickEnvelope]:
        envelopes: list[TickEnvelope] = []
        for phase in self.phase_order:
            for rank in range(world_size):
                envelopes.append(
                    TickEnvelope(tick=tick, phase=phase, src_rank=-1, dst_rank=rank)
                )
        return envelopes


class DistributedWorker:
    def __init__(self, rank_config: RankConfig, transport: Transport) -> None:
        self.rank_config = rank_config
        self.transport = transport

    def validate_rank(self) -> None:
        if self.rank_config.rank_id != self.transport.rank:
            raise ValueError(
                "RankConfig rank_id does not match transport rank: "
                f"{self.rank_config.rank_id} != {self.transport.rank}"
            )


def _message_tag(*, tick: int, phase: TickPhase, face: Face) -> int:
    return tick * 1000 + _PHASE_ID[phase] * 10 + _FACE_ID[face]


def _cpu_payload(payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().contiguous().to("cpu", non_blocking=False)
        for key, value in payload.items()
    }
