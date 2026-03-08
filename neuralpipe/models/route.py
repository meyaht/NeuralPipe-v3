"""Route, Segment, Waypoint, ElbowFitting dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Waypoint:
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z


@dataclass
class ElbowFitting:
    """LR elbow at a waypoint between two segments."""
    location: Waypoint
    # Rotation: unit vectors of incoming and outgoing run directions
    incoming_dir: tuple[float, float, float]
    outgoing_dir: tuple[float, float, float]
    radius_m: float  # centerline bend radius


@dataclass
class SupportPoint:
    location: Waypoint
    steel_member_id: str


@dataclass
class Segment:
    start: Waypoint
    end: Waypoint
    is_grade: bool = False

    @property
    def length_m(self) -> float:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        dz = self.end.z - self.start.z
        return (dx**2 + dy**2 + dz**2) ** 0.5


@dataclass
class Route:
    route_id: str
    line_number: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "VALID"  # VALID | FAILED | FLAGGED
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    waypoints: list[Waypoint] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    elbow_locations: list[ElbowFitting] = field(default_factory=list)
    support_points: list[SupportPoint] = field(default_factory=list)
    eod_m: float = 0.0
    grade_segment_indices: list[int] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    failure_reason: Optional[str] = None

    @property
    def total_length_m(self) -> float:
        return sum(s.length_m for s in self.segments)

    @property
    def num_elbows(self) -> int:
        return len(self.elbow_locations)

    @property
    def num_supports(self) -> int:
        return len(self.support_points)
