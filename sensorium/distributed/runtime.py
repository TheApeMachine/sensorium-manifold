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
            if neighbor == self.rank:
                recv_buffers[face] = send_tensor.clone()
                continue
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
        comm_device = self._comm_device_for_backend()
        received: dict[Face, dict[str, torch.Tensor]] = {}
        send_payloads: dict[Face, dict[str, torch.Tensor]] = {}
        send_counts: dict[Face, int] = {}
        recv_counts: dict[Face, torch.Tensor] = {}
        count_reqs: list[dist.Work] = []

        for face in FACES:
            neighbor = neighbors.get(face, -1)
            if neighbor < 0:
                continue
            send_payload = _normalize_particle_payload(
                payloads.get(face, {}), device=comm_device
            )
            send_payloads[face] = send_payload
            send_n = int(send_payload["positions"].shape[0])
            send_counts[face] = send_n
            if neighbor == self.rank:
                received[face] = {
                    key: value.to(device, non_blocking=False)
                    for key, value in send_payload.items()
                }
                continue

            send_count = torch.tensor(
                [send_n],
                dtype=torch.int64,
                device=comm_device,
            )
            recv_count = torch.zeros((1,), dtype=torch.int64, device=comm_device)
            recv_counts[face] = recv_count
            tag_send = _particle_message_tag(tick=tick, face=face, kind=1)
            tag_recv = _particle_message_tag(
                tick=tick,
                face=OPPOSITE_FACE[face],
                kind=1,
            )
            send_req = dist.isend(
                send_count,
                dst=neighbor,
                tag=tag_send,
                group=self._group,
            )
            recv_req = dist.irecv(
                recv_count,
                src=neighbor,
                tag=tag_recv,
                group=self._group,
            )
            if send_req is not None:
                count_reqs.append(send_req)
            if recv_req is not None:
                count_reqs.append(recv_req)

        for req in count_reqs:
            req.wait()

        data_reqs: list[dist.Work] = []
        recv_vecs: dict[Face, torch.Tensor] = {}
        recv_scas: dict[Face, torch.Tensor] = {}

        for face in FACES:
            neighbor = neighbors.get(face, -1)
            if neighbor < 0 or neighbor == self.rank:
                continue
            send_payload = send_payloads[face]
            send_n = send_counts[face]
            recv_n = int(recv_counts[face].item())

            send_vec, send_sca = _pack_particle_payload(
                send_payload, device=comm_device
            )
            recv_vec = torch.empty((recv_n, 6), dtype=torch.float32, device=comm_device)
            recv_sca = torch.empty((recv_n, 5), dtype=torch.float32, device=comm_device)
            recv_vecs[face] = recv_vec
            recv_scas[face] = recv_sca

            if send_n > 0:
                tag_vec_send = _particle_message_tag(tick=tick, face=face, kind=2)
                tag_sca_send = _particle_message_tag(tick=tick, face=face, kind=3)
                req_sv = dist.isend(
                    send_vec,
                    dst=neighbor,
                    tag=tag_vec_send,
                    group=self._group,
                )
                req_ss = dist.isend(
                    send_sca,
                    dst=neighbor,
                    tag=tag_sca_send,
                    group=self._group,
                )
                if req_sv is not None:
                    data_reqs.append(req_sv)
                if req_ss is not None:
                    data_reqs.append(req_ss)

            if recv_n > 0:
                tag_vec_recv = _particle_message_tag(
                    tick=tick,
                    face=OPPOSITE_FACE[face],
                    kind=2,
                )
                tag_sca_recv = _particle_message_tag(
                    tick=tick,
                    face=OPPOSITE_FACE[face],
                    kind=3,
                )
                req_rv = dist.irecv(
                    recv_vec,
                    src=neighbor,
                    tag=tag_vec_recv,
                    group=self._group,
                )
                req_rs = dist.irecv(
                    recv_sca,
                    src=neighbor,
                    tag=tag_sca_recv,
                    group=self._group,
                )
                if req_rv is not None:
                    data_reqs.append(req_rv)
                if req_rs is not None:
                    data_reqs.append(req_rs)

        for req in data_reqs:
            req.wait()

        for face in FACES:
            if face in received:
                continue
            if face not in recv_vecs:
                continue
            recv_payload = _unpack_particle_payload(recv_vecs[face], recv_scas[face])
            received[face] = {
                key: value.to(device, non_blocking=False)
                for key, value in recv_payload.items()
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
        try:
            dist.all_to_all_single(
                recv_counts_dev,
                send_counts_dev,
                group=self._group,
            )
        except Exception:
            return self._route_mode_payloads_object(payloads_by_rank, device)
        recv_counts = recv_counts_dev.to("cpu")
        send_splits = [int(send_counts[r].item()) for r in range(self.world_size)]
        recv_splits = [int(recv_counts[r].item()) for r in range(self.world_size)]

        send_idx_parts: list[torch.Tensor] = []
        send_real_parts: list[torch.Tensor] = []
        send_imag_parts: list[torch.Tensor] = []
        for dst in range(self.world_size):
            n = send_splits[dst]
            payload = normalized.get(dst)
            if n <= 0 or payload is None:
                send_idx_parts.append(
                    torch.empty((0,), dtype=torch.int64, device=comm_device)
                )
                send_real_parts.append(
                    torch.empty((0,), dtype=torch.float32, device=comm_device)
                )
                send_imag_parts.append(
                    torch.empty((0,), dtype=torch.float32, device=comm_device)
                )
                continue
            send_idx_parts.append(
                payload["mode_idx"].to(comm_device, non_blocking=False)
            )
            send_real_parts.append(payload["real"].to(comm_device, non_blocking=False))
            send_imag_parts.append(payload["imag"].to(comm_device, non_blocking=False))

        send_idx = (
            torch.cat(send_idx_parts, dim=0)
            if send_idx_parts
            else torch.empty((0,), dtype=torch.int64, device=comm_device)
        )
        send_real = (
            torch.cat(send_real_parts, dim=0)
            if send_real_parts
            else torch.empty((0,), dtype=torch.float32, device=comm_device)
        )
        send_imag = (
            torch.cat(send_imag_parts, dim=0)
            if send_imag_parts
            else torch.empty((0,), dtype=torch.float32, device=comm_device)
        )

        recv_total = int(sum(recv_splits))
        recv_idx = torch.empty((recv_total,), dtype=torch.int64, device=comm_device)
        recv_real = torch.empty((recv_total,), dtype=torch.float32, device=comm_device)
        recv_imag = torch.empty((recv_total,), dtype=torch.float32, device=comm_device)

        try:
            dist.all_to_all_single(
                recv_idx,
                send_idx,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=self._group,
            )
            dist.all_to_all_single(
                recv_real,
                send_real,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=self._group,
            )
            dist.all_to_all_single(
                recv_imag,
                send_imag,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=self._group,
            )
        except Exception:
            return self._route_mode_payloads_object(payloads_by_rank, device)

        inbound: list[dict[str, torch.Tensor]] = []
        offset = 0
        for src in range(self.world_size):
            n = recv_splits[src]
            if n <= 0:
                continue
            s = slice(offset, offset + n)
            inbound.append(
                {
                    "mode_idx": recv_idx[s].to(device, non_blocking=False),
                    "real": recv_real[s].to(device, non_blocking=False),
                    "imag": recv_imag[s].to(device, non_blocking=False),
                }
            )
            offset += n
        return inbound

    def _route_mode_payloads_object(
        self,
        payloads_by_rank: dict[int, dict[str, torch.Tensor]],
        device: torch.device,
    ) -> list[dict[str, torch.Tensor]]:
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

    def allreduce_tensor_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        src = tensor.contiguous()
        comm_device = self._comm_device_for_backend()
        if self._use_direct_device_transfer(src):
            out = src.clone()
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self._group)
            return out
        work = (
            src
            if src.device == comm_device
            else src.to(comm_device, non_blocking=False)
        )
        dist.all_reduce(work, op=dist.ReduceOp.SUM, group=self._group)
        return (
            work
            if work.device == src.device
            else work.to(src.device, non_blocking=False)
        )

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


def _particle_message_tag(*, tick: int, face: Face, kind: int) -> int:
    base = _message_tag(tick=tick, phase="particle_migration", face=face)
    return base * 10 + int(kind)


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
        out[key] = src.detach().contiguous().to(device, non_blocking=False)
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


def _pack_particle_payload(
    payload: dict[str, torch.Tensor], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(payload["positions"].shape[0])
    vec = torch.empty((n, 6), dtype=torch.float32, device=device)
    sca = torch.empty((n, 5), dtype=torch.float32, device=device)
    if n == 0:
        return vec, sca
    vec[:, 0:3].copy_(payload["positions"])
    vec[:, 3:6].copy_(payload["velocities"])
    sca[:, 0].copy_(payload["masses"])
    sca[:, 1].copy_(payload["heats"])
    sca[:, 2].copy_(payload["energies"])
    sca[:, 3].copy_(payload["excitations"])
    sca[:, 4].copy_(payload["phase"])
    return vec, sca


def _unpack_particle_payload(
    vec: torch.Tensor, sca: torch.Tensor
) -> dict[str, torch.Tensor]:
    n = int(vec.shape[0])
    if n == 0:
        return {
            "positions": torch.empty((0, 3), dtype=torch.float32, device=vec.device),
            "velocities": torch.empty((0, 3), dtype=torch.float32, device=vec.device),
            "masses": torch.empty((0,), dtype=torch.float32, device=vec.device),
            "heats": torch.empty((0,), dtype=torch.float32, device=vec.device),
            "energies": torch.empty((0,), dtype=torch.float32, device=vec.device),
            "excitations": torch.empty((0,), dtype=torch.float32, device=vec.device),
            "phase": torch.empty((0,), dtype=torch.float32, device=vec.device),
        }
    return {
        "positions": vec[:, 0:3].contiguous(),
        "velocities": vec[:, 3:6].contiguous(),
        "masses": sca[:, 0].contiguous(),
        "heats": sca[:, 1].contiguous(),
        "energies": sca[:, 2].contiguous(),
        "excitations": sca[:, 3].contiguous(),
        "phase": sca[:, 4].contiguous(),
    }
