"""EODClearanceConstraint — obstacle clearance checks per spec Section 2 and 6."""
from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import numpy as np

from .base import AbstractConstraint, ConstraintResult

if TYPE_CHECKING:
    from neuralpipe.models.route import Route

# Spec-defined clearances (metres)
SOLID_OBSTACLE_CLEARANCE_M = 0.050   # 50 mm
ADJACENT_PIPE_CLEARANCE_M = 0.025    # 25 mm
SURFACE_CLEARANCE_M = 0.203          # 8 inches above paving/grating


def _segment_to_point_distance(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    pt: tuple[float, float, float],
) -> float:
    """Minimum distance from point pt to line segment p1-p2."""
    a = np.array(p1, dtype=float)
    b = np.array(p2, dtype=float)
    p = np.array(pt, dtype=float)
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab)) / ab_len_sq
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


class EODClearanceConstraint(AbstractConstraint):
    """Check that pipe + EOD + clearance doesn't clash with solid obstacles.

    Uses the VoxelGridProvider from context. Each occupied voxel is treated
    as a point obstacle; the minimum distance from the pipe segment centreline
    to that voxel centre must be ≥ EOD/2 + solid_clearance.

    Fine clearance checking is spec-mandated from pass 8 onwards.
    """

    name = "eod_clearance"
    scope = "GLOBAL"
    version = "1.0"
    priority = 50

    def check(self, route: "Route", context: dict[str, Any]) -> ConstraintResult:
        # Only enforce from resolution pass 8 onwards (spec §5)
        if context.get("resolution_pass", 8) < 8:
            return ConstraintResult(passed=True, constraint_name=self.name, message="Skipped (pre-pass-8)")

        voxel_grid = context.get("voxel_grid")
        eod_m = route.eod_m
        required_clearance = eod_m / 2 + SOLID_OBSTACLE_CLEARANCE_M

        clashes = []
        for i, seg in enumerate(route.segments):
            p1 = seg.start.as_tuple()
            p2 = seg.end.as_tuple()
            # Build AABB around segment + required_clearance for grid query
            xs = [p1[0], p2[0]]
            ys = [p1[1], p2[1]]
            zs = [p1[2], p2[2]]
            bounds = (
                min(xs) - required_clearance,
                min(ys) - required_clearance,
                min(zs) - required_clearance,
                max(xs) + required_clearance,
                max(ys) + required_clearance,
                max(zs) + required_clearance,
            )
            if voxel_grid is not None:
                occupied = voxel_grid.get_occupied_in_bounds(bounds)
            else:
                occupied = []

            for occ_xyz in occupied:
                dist = _segment_to_point_distance(p1, p2, occ_xyz)
                if dist < required_clearance:
                    clashes.append({
                        "segment_index": i,
                        "obstacle_xyz": occ_xyz,
                        "distance_m": round(dist, 4),
                        "required_m": round(required_clearance, 4),
                    })

        if clashes:
            return ConstraintResult(
                passed=False,
                constraint_name=self.name,
                message=f"{len(clashes)} EOD clearance clash(es) detected.",
                detail={"clashes": clashes},
            )
        return ConstraintResult(passed=True, constraint_name=self.name)


class SurfaceClearanceConstraint(AbstractConstraint):
    """Check minimum vertical clearance above paving/grating surfaces.

    Spec §6: min_centerline_elevation = surface_high_point + 203mm + (EOD/2)

    In v1, surfaces are declared in context["surfaces"] as a list of dicts:
        {"z_high_point_m": float, "label": str}
    The constraint checks that every waypoint's z ≥ the required elevation
    for the highest surface directly below it (we use the global highest surface
    as conservative v1 approximation).
    """

    name = "surface_clearance"
    scope = "GLOBAL"
    version = "1.0"
    priority = 55

    def check(self, route: "Route", context: dict[str, Any]) -> ConstraintResult:
        surfaces = context.get("surfaces", [])
        if not surfaces:
            return ConstraintResult(passed=True, constraint_name=self.name, message="No surfaces defined — skipped")

        eod_m = route.eod_m
        violations = []

        for surface in surfaces:
            z_high = surface.get("z_high_point_m", 0.0)
            min_cl_z = z_high + SURFACE_CLEARANCE_M + eod_m / 2

            for i, wp in enumerate(route.waypoints):
                if wp.z < min_cl_z:
                    violations.append({
                        "waypoint_index": i,
                        "waypoint_z_m": wp.z,
                        "min_required_z_m": round(min_cl_z, 4),
                        "surface_label": surface.get("label", ""),
                    })

        if violations:
            return ConstraintResult(
                passed=False,
                constraint_name=self.name,
                message=f"{len(violations)} waypoint(s) below minimum surface clearance.",
                detail={"violations": violations},
            )
        return ConstraintResult(passed=True, constraint_name=self.name)
