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

_PARTICLE_FIELDS: tuple[str, ...] = (
    "positions",
    "velocities",
    "masses",
    "heats",
    "energies",
    "excitations",
    "phase",
)


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

    @abstractmethod
    def allreduce_tensor_sum(self, tensor: torch.Tensor) -> torch.Tensor:
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

    def allreduce_tensor_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clone()


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
            send_payload = _normalize_particle_payload(
                payloads.get(face, {}), device=device
            )
            send_n = int(send_payload["positions"].shape[0])

            send_count = torch.tensor([send_n], dtype=torch.int64)
            recv_count = torch.zeros((1,), dtype=torch.int64)

            if self.rank < neighbor:
                dist.send(send_count, dst=neighbor, group=self._group)
                dist.recv(recv_count, src=neighbor, group=self._group)
            else:
                dist.recv(recv_count, src=neighbor, group=self._group)
                dist.send(send_count, dst=neighbor, group=self._group)

            recv_n = int(recv_count.item())
            recv_payload_cpu = _empty_particle_payload_cpu(recv_n)

            for key in _PARTICLE_FIELDS:
                send_tensor = send_payload[key]
                recv_tensor = recv_payload_cpu[key]
                if self.rank < neighbor:
                    if send_n > 0:
                        dist.send(send_tensor, dst=neighbor, group=self._group)
                    if recv_n > 0:
                        dist.recv(recv_tensor, src=neighbor, group=self._group)
                else:
                    if recv_n > 0:
                        dist.recv(recv_tensor, src=neighbor, group=self._group)
                    if send_n > 0:
                        dist.send(send_tensor, dst=neighbor, group=self._group)

            received[face] = {
                key: value.to(device, non_blocking=False)
                for key, value in recv_payload_cpu.items()
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
        mode_fields: tuple[str, ...] = ("mode_idx", "real", "imag")
        send_counts = torch.zeros((self.world_size,), dtype=torch.int64)
        normalized: dict[int, dict[str, torch.Tensor]] = {}
        for dst, payload in payloads_by_rank.items():
            if not payload:
                continue
            mode_idx = payload.get("mode_idx")
            real = payload.get("real")
            imag = payload.get("imag")
            if mode_idx is None or real is None or imag is None:
                continue
            n = int(mode_idx.numel())
            if n == 0:
                continue
            send_counts[dst] = n
            normalized[dst] = {
                "mode_idx": mode_idx.detach().contiguous(),
                "real": real.detach().contiguous(),
                "imag": imag.detach().contiguous(),
            }

        comm_device = self._comm_device_for_backend()
        send_counts_dev = send_counts.to(comm_device)
        recv_counts_dev = torch.zeros_like(send_counts_dev)
        dist.all_to_all_single(
            recv_counts_dev,
            send_counts_dev,
            group=self._group,
        )
        recv_counts = recv_counts_dev.to("cpu")

        inbound: list[dict[str, torch.Tensor]] = []
        self_payload = normalized.get(self.rank)
        if self_payload is not None:
            inbound.append(
                {k: v.to(device, non_blocking=False) for k, v in self_payload.items()}
            )

        for peer in range(self.world_size):
            if peer == self.rank:
                continue
            send_n = int(send_counts[peer].item())
            recv_n = int(recv_counts[peer].item())
            recv_payload: dict[str, torch.Tensor] = {}
            send_payload = normalized.get(peer)

            for key in mode_fields:
                send_tensor = None
                if send_n > 0 and send_payload is not None:
                    send_tensor = send_payload[key].to(comm_device, non_blocking=False)

                recv_tensor = None
                if recv_n > 0:
                    dtype = torch.int64 if key == "mode_idx" else torch.float32
                    recv_tensor = torch.empty(
                        (recv_n,), dtype=dtype, device=comm_device
                    )

                if self.rank < peer:
                    if send_tensor is not None:
                        dist.send(send_tensor, dst=peer, group=self._group)
                    if recv_tensor is not None:
                        dist.recv(recv_tensor, src=peer, group=self._group)
                else:
                    if recv_tensor is not None:
                        dist.recv(recv_tensor, src=peer, group=self._group)
                    if send_tensor is not None:
                        dist.send(send_tensor, dst=peer, group=self._group)

                if recv_tensor is not None:
                    recv_payload[key] = recv_tensor.to(device, non_blocking=False)

            if recv_n > 0:
                inbound.append(recv_payload)

        return inbound

    def allreduce_tensor_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        src = tensor.contiguous()
        if self._use_direct_device_transfer(src):
            out = src.clone()
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self._group)
            return out
        cpu = src.to("cpu", non_blocking=False)
        dist.all_reduce(cpu, op=dist.ReduceOp.SUM, group=self._group)
        return cpu.to(src.device, non_blocking=False)

    def _use_direct_device_transfer(self, tensor: torch.Tensor) -> bool:
        backend = str(self._backend).lower()
        return tensor.device.type == "cuda" and backend == "nccl"

    def _comm_device_for_backend(self) -> torch.device:
        backend = str(self._backend).lower()
        if backend == "nccl" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


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


def _normalize_particle_payload(
    payload: dict[str, torch.Tensor], *, device: torch.device
) -> dict[str, torch.Tensor]:
    pos = payload.get("positions")
    n = int(pos.shape[0]) if pos is not None else 0
    out = _empty_particle_payload_cpu(n)
    if n == 0:
        return out

    for key in _PARTICLE_FIELDS:
        src = payload.get(key)
        if src is None:
            continue
        out[key] = src.detach().contiguous().to("cpu", non_blocking=False)
    del device
    return out


def _empty_particle_payload_cpu(n: int) -> dict[str, torch.Tensor]:
    return {
        "positions": torch.empty((n, 3), dtype=torch.float32),
        "velocities": torch.empty((n, 3), dtype=torch.float32),
        "masses": torch.empty((n,), dtype=torch.float32),
        "heats": torch.empty((n,), dtype=torch.float32),
        "energies": torch.empty((n,), dtype=torch.float32),
        "excitations": torch.empty((n,), dtype=torch.float32),
        "phase": torch.empty((n,), dtype=torch.float32),
    }
