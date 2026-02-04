from __future__ import annotations

import csv
import json
import os
from typing import Any, Optional


class BridgeDiagnosticsLogger:
    """Write BridgeObserveOutput metrics to CSV and/or JSONL."""

    def __init__(self, *, csv_path: Optional[str] = None, jsonl_path: Optional[str] = None):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path

    def log(self, *, step: int, out: Any) -> None:
        record = {
            "step": int(step),
            "mismatch_mean": float(out.mismatch_mean),
            "mismatch_min": float(out.mismatch_min),
            "mismatch_max": float(out.mismatch_max),
            "sem_entropy": float(out.sem_entropy),
            "spec_entropy": float(out.spec_entropy),
            "ratio": float(out.ratio),
            "heat_mean": float(out.heat_mean),
            "energy_mean": float(out.energy_mean),
        }

        if self.jsonl_path:
            os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            file_exists = os.path.exists(self.csv_path)
            with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(record.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)


class SemanticDiagnosticsLogger:
    """Write semantic pondering metrics/events to CSV and/or JSONL."""

    def __init__(self, *, csv_path: Optional[str] = None, jsonl_path: Optional[str] = None):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path

    def log(self, record: dict[str, Any]) -> None:
        if self.jsonl_path:
            os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            file_exists = os.path.exists(self.csv_path)
            with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(record.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)
