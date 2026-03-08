"""OrthogonalityConstraint — all segments must be parallel/perpendicular to XYZ axes."""
from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

from .base import AbstractConstraint, ConstraintResult
from neuralpipe.models.route import Waypoint

if TYPE_CHECKING:
    from neuralpipe.models.route import Route

# Tolerance in degrees for "close enough to orthogonal"
_ORTHOGONAL_TOL_DEG = 0.5


def _is_orthogonal(p1: Waypoint, p2: Waypoint) -> bool:
    """True if the segment p1→p2 is parallel to one of the three global axes."""
    dx = abs(p2.x - p1.x)
    dy = abs(p2.y - p1.y)
    dz = abs(p2.z - p1.z)
    length = math.sqrt(dx**2 + dy**2 + dz**2)
    if length < 1e-6:
        return True  # degenerate segment — treat as passing
    # Exactly one axis component should dominate; the other two should be near zero
    comps = [dx, dy, dz]
    comps.sort(reverse=True)
    # The largest component / length should be ~1.0; the rest ~0.0
    cos_angle = comps[0] / length
    return cos_angle >= math.cos(math.radians(_ORTHOGONAL_TOL_DEG))


class OrthogonalityConstraint(AbstractConstraint):
    """All route segments must be parallel or perpendicular to the global XYZ axes.

    Per spec Section 7, Step 2: orthogonality is not enforced during coarse
    passes 1–4. The routing engine sets `coarse_pass` in context to skip this
    constraint during early resolution passes.
    """

    name = "orthogonality"
    scope = "GLOBAL"
    version = "1.0"
    priority = 20

    def check(self, route: "Route", context: dict[str, Any]) -> ConstraintResult:
        # Skip during coarse passes 1-4
        if context.get("resolution_pass", 5) <= 4:
            return ConstraintResult(passed=True, constraint_name=self.name, message="Skipped (coarse pass)")

        failing_segments = []
        for i, seg in enumerate(route.segments):
            if not _is_orthogonal(seg.start, seg.end):
                failing_segments.append(i)

        if failing_segments:
            return ConstraintResult(
                passed=False,
                constraint_name=self.name,
                message=f"Segments {failing_segments} are not orthogonal to global axes.",
                detail={"failing_segment_indices": failing_segments},
            )
        return ConstraintResult(passed=True, constraint_name=self.name)

    def repair(self, route: "Route", context: dict[str, Any]):
        """Snap non-orthogonal segments by inserting a midpoint vertex.

        For each failing segment, we split it at the midpoint and adjust the
        midpoint so both resulting sub-segments are axis-aligned.
        """
        from neuralpipe.models.route import Route, Segment, Waypoint

        new_waypoints = list(route.waypoints)
        new_segments: list[Segment] = []

        for seg in route.segments:
            if _is_orthogonal(seg.start, seg.end):
                new_segments.append(seg)
                continue
            # Insert orthogonal midpoint: go in X first, then Y or Z
            mid = Waypoint(x=seg.end.x, y=seg.start.y, z=seg.start.z)
            new_segments.append(Segment(start=seg.start, end=mid, is_grade=seg.is_grade))
            new_segments.append(Segment(start=mid, end=seg.end, is_grade=seg.is_grade))
            new_waypoints.append(mid)

        repaired = Route(
            route_id=route.route_id,
            line_number=route.line_number,
            status=route.status,
            eod_m=route.eod_m,
            waypoints=[new_segments[0].start] + [s.end for s in new_segments],
            segments=new_segments,
            elbow_locations=route.elbow_locations,
            support_points=route.support_points,
            grade_segment_indices=route.grade_segment_indices,
            flags=route.flags,
        )
        return repaired
