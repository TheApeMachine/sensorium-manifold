from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


def _lcp(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    """Longest common prefix length for two tuples."""
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

    def __init__(self):
        self.value: Optional[int] = None
        # Keyed by first token of edge.label for fast selection.
        self.edges: Dict[int, _Edge] = {}


class RadixTrie:
    """Radix trie mapping token sequences -> stable IDs.

    This is a path-compressed trie:
    - Each edge stores a tuple label (a run of tokens).
    - Insert/lookup are O(L) in sequence length with low constant factors.
    """

    def __init__(self):
        self._root = _Node()

    def get(self, seq: Tuple[int, ...]) -> Optional[int]:
        node = self._root
        rest = seq
        while rest:
            e = node.edges.get(rest[0])
            if e is None:
                return None
            lab = e.label
            if len(rest) < len(lab) or rest[: len(lab)] != lab:
                return None
            rest = rest[len(lab) :]
            node = e.child
        return node.value

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

            # Full edge match, continue.
            if i == len(lab):
                node = e.child
                rest = rest[i:]
                continue

            # Need to split the existing edge at i.
            # Create intermediate node for the common prefix.
            common = lab[:i]
            old_suffix = lab[i:]
            new_suffix = rest[i:]

            mid = _Node()
            # Replace edge with common prefix -> mid
            node.edges[first] = _Edge(label=common, child=mid)

            # Reattach old edge suffix
            mid.edges[old_suffix[0]] = _Edge(label=old_suffix, child=e.child)

            if not new_suffix:
                # New sequence ends at the split.
                mid.value = int(value)
                return

            # Attach new suffix as fresh edge
            new_child = _Node()
            new_child.value = int(value)
            mid.edges[new_suffix[0]] = _Edge(label=new_suffix, child=new_child)
            return

        node.value = int(value)

