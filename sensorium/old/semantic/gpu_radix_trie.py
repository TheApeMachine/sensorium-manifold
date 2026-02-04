from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


def _lcp(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class _Edge:
    label: Tuple[int, ...]
    child: "_Node"


class _Node:
    __slots__ = ("value", "edges")

    def __init__(self) -> None:
        self.value: Optional[int] = None
        self.edges: Dict[int, _Edge] = {}


class _RadixTrieBuilder:
    def __init__(self) -> None:
        self._root = _Node()

    def insert(self, seq: Tuple[int, ...], value: int) -> None:
        if not seq:
            self._root.value = int(value)
            return
        node = self._root
        rest = seq
        while rest:
            first = rest[0]
            e = node.edges.get(first)
            if e is None:
                child = _Node()
                child.value = int(value)
                node.edges[first] = _Edge(label=rest, child=child)
                return

            lab = e.label
            i = _lcp(rest, lab)

            if i == len(lab):
                node = e.child
                rest = rest[i:]
                continue

            common = lab[:i]
            old_suffix = lab[i:]
            new_suffix = rest[i:]

            mid = _Node()
            node.edges[first] = _Edge(label=common, child=mid)

            mid.edges[old_suffix[0]] = _Edge(label=old_suffix, child=e.child)

            if not new_suffix:
                mid.value = int(value)
                return

            new_child = _Node()
            new_child.value = int(value)
            mid.edges[new_suffix[0]] = _Edge(label=new_suffix, child=new_child)
            return

        node.value = int(value)

    @property
    def root(self) -> _Node:
        return self._root


class GpuRadixTrie:
    def __init__(self, *, device: torch.device, pad_id: int) -> None:
        self.device = device
        self.pad_id = int(pad_id)
        self.clear()

    def clear(self) -> None:
        self.node_value = torch.empty(0, device=self.device, dtype=torch.long)
        self.node_edge_start = torch.empty(0, device=self.device, dtype=torch.long)
        self.node_edge_count = torch.empty(0, device=self.device, dtype=torch.long)
        self.edge_child = torch.empty(0, device=self.device, dtype=torch.long)
        self.edge_label_len = torch.empty(0, device=self.device, dtype=torch.long)
        self.edge_first_token = torch.empty(0, device=self.device, dtype=torch.long)
        self.edge_label_pad = torch.empty(0, 0, device=self.device, dtype=torch.long)
        self.max_label_len = 0
        self.max_depth = 0

    def rebuild(self, seq_pad: torch.Tensor, values: torch.Tensor) -> None:
        """Rebuild packed radix trie from padded sequences.

        seq_pad: [N, order], padded with pad_id.
        values: [N] value for each sequence.
        """
        if seq_pad.numel() == 0:
            self.clear()
            return

        seq_cpu = seq_pad.detach().to("cpu")
        val_cpu = values.detach().to("cpu")
        builder = _RadixTrieBuilder()
        max_depth = 0
        for row, v in zip(seq_cpu.tolist(), val_cpu.tolist()):
            if self.pad_id in row:
                l = row.index(self.pad_id)
            else:
                l = len(row)
            max_depth = max(max_depth, l)
            builder.insert(tuple(int(x) for x in row[:l]), int(v))

        # Assign node indices.
        nodes: list[_Node] = []
        node_to_idx: Dict[_Node, int] = {}

        def assign(node: _Node) -> None:
            node_to_idx[node] = len(nodes)
            nodes.append(node)
            for e in node.edges.values():
                if e.child not in node_to_idx:
                    assign(e.child)

        assign(builder.root)

        # Pack edges.
        edge_child: list[int] = []
        edge_label_len: list[int] = []
        edge_first_token: list[int] = []
        edge_labels: list[Tuple[int, ...]] = []
        node_edge_start: list[int] = []
        node_edge_count: list[int] = []
        node_value: list[int] = []

        for node in nodes:
            node_value.append(int(node.value) if node.value is not None else -1)
            node_edge_start.append(len(edge_child))
            edges = list(node.edges.values())
            node_edge_count.append(len(edges))
            for e in edges:
                edge_child.append(node_to_idx[e.child])
                edge_label_len.append(len(e.label))
                edge_first_token.append(int(e.label[0]))
                edge_labels.append(e.label)

        max_label_len = max(edge_label_len) if edge_label_len else 0
        if max_label_len == 0:
            max_label_len = 1
            edge_labels = [(self.pad_id,) for _ in edge_labels]

        label_pad: list[list[int]] = []
        for lab in edge_labels:
            padded = list(lab) + [self.pad_id] * (max_label_len - len(lab))
            label_pad.append(padded)

        self.node_value = torch.tensor(node_value, device=self.device, dtype=torch.long)
        self.node_edge_start = torch.tensor(node_edge_start, device=self.device, dtype=torch.long)
        self.node_edge_count = torch.tensor(node_edge_count, device=self.device, dtype=torch.long)
        self.edge_child = torch.tensor(edge_child, device=self.device, dtype=torch.long)
        self.edge_label_len = torch.tensor(edge_label_len, device=self.device, dtype=torch.long)
        self.edge_first_token = torch.tensor(edge_first_token, device=self.device, dtype=torch.long)
        if label_pad:
            self.edge_label_pad = torch.tensor(label_pad, device=self.device, dtype=torch.long)
        else:
            self.edge_label_pad = torch.empty(0, max_label_len, device=self.device, dtype=torch.long)
        self.max_label_len = int(max_label_len)
        self.max_depth = int(max_depth)

    def lookup(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = seq.to(device=self.device, dtype=torch.long)
        if seq.ndim == 1:
            seq = seq.view(1, -1)
        if self.node_value.numel() == 0:
            idx = torch.full((int(seq.shape[0]),), -1, device=self.device, dtype=torch.long)
            exists = torch.zeros((int(seq.shape[0]),), device=self.device, dtype=torch.bool)
            return idx, exists

        n, L = int(seq.shape[0]), int(seq.shape[1])
        node = torch.zeros(n, device=self.device, dtype=torch.long)
        pos = torch.zeros(n, device=self.device, dtype=torch.long)
        alive = torch.ones(n, device=self.device, dtype=torch.bool)
        idx = torch.full((n,), -1, device=self.device, dtype=torch.long)
        exists = torch.zeros((n,), device=self.device, dtype=torch.bool)

        max_label_len = int(self.max_label_len)
        arange = torch.arange(max_label_len, device=self.device)

        steps = 0
        max_steps = max(1, int(self.max_depth) + 1)
        while bool(alive.any()) and steps < max_steps:
            steps += 1
            active = alive.nonzero(as_tuple=False).flatten()
            if active.numel() == 0:
                break

            node_a = node[active]
            pos_a = pos[active]
            done = pos_a >= L
            if bool(done.any()):
                node_done = node_a[done]
                val = self.node_value[node_done]
                idx[active[done]] = val
                exists[active[done]] = val >= 0
                alive[active[done]] = False
                keep = ~done
                active = active[keep]
                if active.numel() == 0:
                    break
                node_a = node_a[keep]
                pos_a = pos_a[keep]

            edge_start = self.node_edge_start[node_a]
            edge_count = self.node_edge_count[node_a]
            no_edge = edge_count == 0
            if bool(no_edge.any()):
                alive[active[no_edge]] = False
                keep = ~no_edge
                active = active[keep]
                if active.numel() == 0:
                    break
                node_a = node_a[keep]
                pos_a = pos_a[keep]
                edge_start = edge_start[keep]
                edge_count = edge_count[keep]

            num_active = int(active.numel())
            total_edges = int(edge_count.sum().item())
            if total_edges == 0:
                alive[active] = False
                break

            group_ids = torch.repeat_interleave(torch.arange(num_active, device=self.device), edge_count)
            prefix = torch.cumsum(edge_count, dim=0)
            start_offsets = torch.zeros_like(prefix)
            start_offsets[1:] = prefix[:-1]
            local = torch.arange(total_edges, device=self.device) - start_offsets[group_ids]
            edge_idx = edge_start[group_ids] + local

            tok = seq[active, pos_a]
            edge_first = self.edge_first_token[edge_idx]
            match_first = edge_first == tok[group_ids]

            edge_idx_first = torch.full((num_active,), -1, device=self.device, dtype=torch.long)
            if bool(match_first.any()):
                edge_idx_first.scatter_(0, group_ids[match_first], edge_idx[match_first])

            has_edge = edge_idx_first >= 0
            if not bool(has_edge.any()):
                alive[active] = False
                continue

            seq_active = seq[active]
            pos_matrix = pos_a[:, None] + arange[None, :]
            in_bounds = pos_matrix < L
            pos_matrix_clamped = pos_matrix.clamp(max=L - 1)
            seq_window = seq_active.gather(1, pos_matrix_clamped)
            if not bool(in_bounds.all()):
                seq_window = torch.where(
                    in_bounds,
                    seq_window,
                    torch.full_like(seq_window, self.pad_id),
                )

            edge_idx_safe = edge_idx_first.clamp(min=0)
            edge_label = self.edge_label_pad[edge_idx_safe]
            label_len = self.edge_label_len[edge_idx_safe]
            label_mask = arange[None, :] < label_len[:, None]
            label_match = ((edge_label == seq_window) | ~label_mask).all(dim=1)
            len_ok = pos_a + label_len <= L
            ok = has_edge & label_match & len_ok

            if bool(ok.any()):
                new_node = self.edge_child[edge_idx_safe[ok]]
                node[active[ok]] = new_node
                pos[active[ok]] = pos_a[ok] + label_len[ok]

            fail = ~ok
            if bool(fail.any()):
                alive[active[fail]] = False

        return idx, exists
