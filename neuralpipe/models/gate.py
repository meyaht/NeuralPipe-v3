"""GateOpening — pipe rack gate openings detected by GateDetector / AutoGateDetector."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GateOpening:
    gate_id:   str
    axis:      str            # 'X' or 'Y'  (slice-normal axis of the gate slab)
    position_m: float         # slice centre position along the axis
    bbox_3d:   list           # [xmin, ymin, zmin, xmax, ymax, zmax]
    confidence: float = 1.0
    pipe_count: int = 0

    def center_3d(self) -> tuple[float, float, float]:
        b = self.bbox_3d
        return ((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2)

    def to_dict(self) -> dict:
        return {
            "gate_id":    self.gate_id,
            "axis":       self.axis,
            "position_m": self.position_m,
            "bbox_3d":    self.bbox_3d,
            "confidence": self.confidence,
            "pipe_count": self.pipe_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GateOpening":
        return cls(
            gate_id=d["gate_id"],
            axis=d["axis"],
            position_m=d["position_m"],
            bbox_3d=d["bbox_3d"],
            confidence=d.get("confidence", 1.0),
            pipe_count=d.get("pipe_count", 0),
        )


def load_gates(path: str | Path) -> list[GateOpening]:
    """Load gates from a GateDetector / AutoGateDetector gates.json file.

    The JSON may be either the raw list saved by GateDetector (list of gate dicts)
    or the AutoGateDetector pipeline format { "gates": [...], ... }.
    """
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        gate_list = raw
    elif isinstance(raw, dict):
        gate_list = raw.get("gates", [])
    else:
        raise ValueError(f"Unrecognised gates.json format in {path}")

    gates = []
    for g in gate_list:
        if "bbox_3d" not in g:
            continue  # skip entries without 3D bounds (shouldn't happen)
        gates.append(GateOpening.from_dict(g))
    return gates
