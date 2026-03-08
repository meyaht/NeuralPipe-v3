"""PipeSpec dataclass and loader from config/pipe_specs.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipeSpec:
    nps_inches: float
    od_mm: float
    preferred_span_mm: float
    max_span_mm: float
    lr_elbow_radius_mm: float

    @property
    def od_m(self) -> float:
        return self.od_mm / 1000.0

    @property
    def preferred_span_m(self) -> float:
        return self.preferred_span_mm / 1000.0

    @property
    def max_span_m(self) -> float:
        return self.max_span_mm / 1000.0

    @property
    def lr_elbow_radius_m(self) -> float:
        return self.lr_elbow_radius_mm / 1000.0


class PipeSpecRegistry:
    """Loads and retrieves PipeSpec entries from config JSON."""

    def __init__(self, config_path: str | Path) -> None:
        self._specs: dict[float, PipeSpec] = {}
        self._load(Path(config_path))

    def _load(self, path: Path) -> None:
        with open(path) as f:
            data = json.load(f)
        for entry in data["nominal_diameters"]:
            spec = PipeSpec(
                nps_inches=entry["nps_inches"],
                od_mm=entry["od_mm"],
                preferred_span_mm=entry["preferred_span_mm"],
                max_span_mm=entry["max_span_mm"],
                lr_elbow_radius_mm=entry["lr_elbow_radius_mm"],
            )
            self._specs[spec.nps_inches] = spec

    def get(self, nps_inches: float) -> PipeSpec | None:
        """Return exact match or None if not found."""
        return self._specs.get(nps_inches)

    def get_or_interpolate(self, nps_inches: float) -> PipeSpec:
        """Return exact match; interpolate between nearest two sizes if not listed."""
        if nps_inches in self._specs:
            return self._specs[nps_inches]
        sizes = sorted(self._specs.keys())
        lower = [s for s in sizes if s < nps_inches]
        upper = [s for s in sizes if s > nps_inches]
        if not lower or not upper:
            raise ValueError(
                f"NPS {nps_inches}\" is outside tabulated range {sizes[0]}\"–{sizes[-1]}\". "
                "User confirmation required."
            )
        lo = self._specs[lower[-1]]
        hi = self._specs[upper[0]]
        t = (nps_inches - lo.nps_inches) / (hi.nps_inches - lo.nps_inches)
        return PipeSpec(
            nps_inches=nps_inches,
            od_mm=lo.od_mm + t * (hi.od_mm - lo.od_mm),
            preferred_span_mm=lo.preferred_span_mm + t * (hi.preferred_span_mm - lo.preferred_span_mm),
            max_span_mm=lo.max_span_mm + t * (hi.max_span_mm - lo.max_span_mm),
            lr_elbow_radius_mm=lo.lr_elbow_radius_mm + t * (hi.lr_elbow_radius_mm - lo.lr_elbow_radius_mm),
        )

    @property
    def all_specs(self) -> list[PipeSpec]:
        return list(self._specs.values())
