"""Universal Tokenizer (modality-agnostic).

Matches paper/main.tex §3: IDs are deterministic hashes of (Byte, Index).

We reserve a small prefix for special tokens, then map hashed IDs into a fixed
vocabulary range. This means *no dataset-specific vocabulary building*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch


@dataclass(frozen=True, slots=True)
class UniversalTokenizerConfig:
    hash_vocab_size: int = 4096
    hash_prime: int = 31
    special_tokens: tuple[str, ...] = ("<pad>", "<unk>", "<bos>", "<eos>", "<mask>")

    # Optional label space for classification tasks.
    num_labels: int = 0


class UniversalTokenizer:
    """Deterministic byte+index hashing tokenizer."""

    def __init__(self, cfg: UniversalTokenizerConfig):
        if int(cfg.hash_vocab_size) <= 0:
            raise ValueError("hash_vocab_size must be > 0")
        self.cfg = cfg
        self.special_size = len(cfg.special_tokens)
        self.hash_vocab_size = int(cfg.hash_vocab_size)
        self.num_labels = int(cfg.num_labels)
        self.vocab_size = self.special_size + self.hash_vocab_size + self.num_labels

        self.pad_id = 0
        self.unk_id = 1 if self.special_size > 1 else 0
        self.bos_id = 2 if self.special_size > 2 else 0
        self.eos_id = 3 if self.special_size > 3 else 0
        self.mask_id = 4 if self.special_size > 4 else 0

        self.label_offset = self.special_size + self.hash_vocab_size

    @property
    def vocab(self) -> List[str]:
        # Only used for manifold compatibility / dashboards.
        toks = list(self.cfg.special_tokens)
        toks += [f"hash_{i}" for i in range(self.hash_vocab_size)]
        toks += [f"label_{i}" for i in range(self.num_labels)]
        return toks

    def label_id(self, label: int) -> int:
        if self.num_labels <= 0:
            raise ValueError("Tokenizer has no labels configured")
        if not (0 <= int(label) < self.num_labels):
            raise ValueError(f"label out of range: {label}")
        return int(self.label_offset + int(label))

    def _hash_byte_pos(self, byte_val: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        b = byte_val.to(torch.long)
        p = pos.to(torch.long)
        hashed = (b * int(self.cfg.hash_prime) + p) % self.hash_vocab_size
        return hashed + self.special_size

    def encode_bytes(
        self,
        data: torch.Tensor,
        *,
        add_bos_eos: bool = False,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Encode a uint8 tensor to token IDs.

        Args:
            data: uint8 tensor (any shape); will be flattened
            add_bos_eos: prepend/append BOS/EOS
            position_offset: added to index before hashing
        """
        if data.dtype != torch.uint8:
            raise TypeError(f"encode_bytes expects uint8, got {data.dtype}")
        flat = data.flatten()
        n = int(flat.numel())
        pos = torch.arange(position_offset, position_offset + n, device=flat.device, dtype=torch.long)
        ids = self._hash_byte_pos(flat, pos).to(torch.long)
        if add_bos_eos:
            ids = torch.cat(
                [
                    torch.tensor([self.bos_id], device=ids.device, dtype=torch.long),
                    ids,
                    torch.tensor([self.eos_id], device=ids.device, dtype=torch.long),
                ],
                dim=0,
            )
        return ids

    def encode_text(
        self,
        text: str,
        *,
        add_bos_eos: bool = False,
        position_offset: int = 0,
        encoding: str = "utf-8",
        errors: str = "ignore",
    ) -> torch.Tensor:
        b = text.encode(encoding, errors=errors)
        t = torch.tensor(list(b), dtype=torch.uint8)
        return self.encode_bytes(t, add_bos_eos=add_bos_eos, position_offset=position_offset)

