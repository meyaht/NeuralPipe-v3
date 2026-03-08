"""ExclusionZoneConstraint — hard exclusion zones per spec Section 6."""
from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import numpy as np

from .base import AbstractConstraint, ConstraintResult

if TYPE_CHECKING:
    from neuralpipe.models.route import Route


# Spec §6: minimum vertical clearances (metres) by zone type
ZONE_CLEARANCES_M: dict[str, dict[str, float]] = {
    "vehicle_roadway": {
        "vertical_m": 4.800,
        "horizontal_buffer_m": 0.600,
    },
    "rail_crossing": {
        "vertical_m": 6.700,
        "horizontal_buffer_m": 3.000,
    },
    "pedestrian_walkway": {
        "vertical_m": 2.200,
        "horizontal_buffer_m": 0.300,
    },
    "crane_path": {
        "vertical_m": 0.0,   # User-defined, handled via user polygon
        "horizontal_buffer_m": 0.0,
    },
    "emergency_egress": {
        "vertical_m": 2.200,
        "horizontal_buffer_m": 0.600,
    },
}

# Proximity to unclassified open area at grade triggers crane path warning
CRANE_PATH_PROXIMITY_WARNING_M = 10.0


def _segment_to_point_xy_distance(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    pt: tuple[float, float, float],
) -> float:
    """XY-plane distance from segment p1-p2 to point pt (ignoring Z)."""
    a = np.array([p1[0], p1[1]], dtype=float)
    b = np.array([p2[0], p2[1]], dtype=float)
    p = np.array([pt[0], pt[1]], dtype=float)
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab)) / ab_len_sq
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


class ExclusionZoneConstraint(AbstractConstraint):
    """Hard-fail any route that enters a declared exclusion zone.

    Exclusion zones are passed in context["exclusion_zones"] as a list of dicts:
    {
        "zone_type": "vehicle_roadway" | "rail_crossing" | "pedestrian_walkway"
                     | "crane_path" | "emergency_egress",
        "centre_xyz": (x, y, z),       # zone reference point (metres)
        "label": str,                  # human description
        # For crane_path: user-defined polygon (list of (x,y) tuples)
        "polygon_xy": [(x1,y1), ...],  # optional
    }

    For each zone, we check whether any route segment passes within the
    defined buffer radius of the zone centre. If the segment's lowest z
    is below the zone's required vertical clearance above the reference z,
    it's a violation.
    """

    name = "exclusion_zone"
    scope = "GLOBAL"
    version = "1.0"
    priority = 10  # Check first — hard fail

    def check(self, route: "Route", context: dict[str, Any]) -> ConstraintResult:
        zones = context.get("exclusion_zones", [])
        eod_m = route.eod_m
        violations = []
        warnings = []

        for zone in zones:
            zone_type = zone.get("zone_type", "crane_path")
            centre = zone.get("centre_xyz", (0.0, 0.0, 0.0))
            label = zone.get("label", zone_type)
            specs = ZONE_CLEARANCES_M.get(zone_type, {"vertical_m": 0.0, "horizontal_buffer_m": 1.0})
            h_buf = specs["horizontal_buffer_m"] + eod_m / 2
            v_cl = specs["vertical_m"]

            for i, seg in enumerate(route.segments):
                p1 = seg.start.as_tuple()
                p2 = seg.end.as_tuple()

                xy_dist = _segment_to_point_xy_distance(p1, p2, centre)

                if xy_dist < h_buf:
                    # Check vertical clearance
                    seg_min_z = min(p1[2], p2[2]) - eod_m / 2
                    zone_ref_z = centre[2]
                    if seg_min_z < zone_ref_z + v_cl:
                        if zone_type == "crane_path":
                            warnings.append({
                                "segment_index": i,
                                "zone_label": label,
                                "xy_distance_m": round(xy_dist, 3),
                            })
                        else:
                            violations.append({
                                "segment_index": i,
                                "zone_type": zone_type,
                                "zone_label": label,
                                "xy_distance_m": round(xy_dist, 3),
                                "seg_min_z_m": round(seg_min_z, 3),
                                "required_z_m": round(zone_ref_z + v_cl, 3),
                            })

        if violations:
            return ConstraintResult(
                passed=False,
                constraint_name=self.name,
                message=f"Route enters {len(violations)} exclusion zone(s).",
                detail={"violations": violations, "warnings": warnings},
            )

        if warnings:
            # Crane path conflict — flagged, not failed
            for warning in warnings:
                route.flags.append(
                    f"Potential Crane Path Conflict - User Review Required: {warning['zone_label']}"
                )
            return ConstraintResult(
                passed=True,
                constraint_name=self.name,
                message=f"{len(warnings)} potential crane path conflict(s) flagged for review.",
                detail={"warnings": warnings},
            )

        return ConstraintResult(passed=True, constraint_name=self.name)
