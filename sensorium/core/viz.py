from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def plot_pondering_jsonl(jsonl_path: Path, out_png: Path) -> Path:
    """Plot pondering diagnostics time series to a PNG."""
    rows = load_jsonl(jsonl_path)
    if not rows:
        return out_png

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return out_png

    x = [int(r.get("step", i)) for i, r in enumerate(rows)]
    shortcuts = [float(r.get("shortcuts", 0.0)) for r in rows]
    dead_ends = [float(r.get("dead_ends", 0.0)) for r in rows]
    hunger = [float(r.get("hunger_mean", 0.0)) for r in rows]
    heat = [float(r.get("heat_mean", 0.0)) for r in rows]
    exc = [float(r.get("exc_mean", 0.0)) for r in rows]
    active = [float(r.get("active", 0.0)) for r in rows]

    cum_short = []
    s = 0.0
    for v in shortcuts:
        s += v
        cum_short.append(s)

    cum_dead = []
    d = 0.0
    for v in dead_ends:
        d += v
        cum_dead.append(d)

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax[0].plot(x, shortcuts, label="shortcuts per step")
    ax[0].plot(x, dead_ends, label="dead_ends per step")
    ax[0].plot(x, active, label="active sources")
    ax[0].set_ylabel("events / locality")
    ax[0].legend(loc="upper left")

    ax[1].plot(x, hunger, label="hunger_mean")
    ax[1].plot(x, heat, label="heat_mean")
    ax[1].plot(x, exc, label="exc_mean")
    ax[1].plot(x, cum_short, label="cumulative shortcuts")
    ax[1].plot(x, cum_dead, label="cumulative dead_ends")
    ax[1].set_ylabel("state / cumulative")
    ax[1].set_xlabel("time step")
    ax[1].legend(loc="upper left")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png

