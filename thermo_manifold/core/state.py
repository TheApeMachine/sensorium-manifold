from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

try:
    import torch
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency: PyTorch (`torch`).\n\n"
        "Install it with one of:\n"
        "- `pip install torch`\n"
        "- `uv pip install torch`\n\n"
        "If you're running from a checkout, also ensure the project deps are installed:\n"
        "- `pip install -e .`\n"
        "- `uv pip install -e .`\n"
    ) from e


@dataclass
class BatchState:
    """A lightweight (TensorDict-free) batched container.

    All tensors are expected to have the same leading dimension `n`.
    """

    data: Dict[str, torch.Tensor]

    @staticmethod
    def empty() -> "BatchState":
        return BatchState({})

    @property
    def n(self) -> int:
        if not self.data:
            return 0
        for v in self.data.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                return int(v.shape[0])
        return 0

    def keys(self) -> Iterable[str]:
        return self.data.keys()

    def has(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        return self.data.get(key, default)

    def set(self, key: str, value: torch.Tensor) -> None:
        self.data[key] = value

    def ensure(self, key: str, shape0: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if key not in self.data:
            self.data[key] = torch.zeros(shape0, device=device, dtype=dtype)
        return self.data[key]

    def select(self, idx: torch.Tensor) -> "BatchState":
        if idx.numel() == 0:
            return BatchState.empty()
        out: Dict[str, torch.Tensor] = {}
        for k, v in self.data.items():
            if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == self.n:
                out[k] = v[idx]
            else:
                out[k] = v
        return BatchState(out)

    def writeback(self, idx: torch.Tensor, subset: "BatchState") -> None:
        for k, v in subset.data.items():
            if k in self.data and isinstance(self.data[k], torch.Tensor) and self.data[k].ndim >= 1 and self.data[k].shape[0] == self.n:
                self.data[k][idx] = v
